import torch.nn as nn
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb
import cupy as cp
CUPY_AVAILABLE = True

def felzenszwalb_img2labels(
    image_path: str,
    scale: int = 40,
    sigma: float = 0.8,
    min_size: int = 20,
):
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    labels = felzenszwalb(img_rgb, scale=scale, sigma=sigma, min_size=min_size)
    uniq = np.unique(labels)
    remap = {int(u): i for i, u in enumerate(uniq)}
    labels = np.vectorize(remap.get)(labels).astype(np.int32)
    return labels
def grad_forward(q):
    gx = q[..., :, 1:] - q[..., :, :-1]
    gy = q[..., 1:, :] - q[..., :-1, :]
    return gx, gy
def div_backward(px, py):
    B, K, H, Wm1 = px.shape
    W = Wm1 + 1
    div = torch.zeros((B, K, H, W), device=px.device, dtype=px.dtype)
    div[..., :, 0] += -px[..., :, 0]
    div[..., :, 1:-1] += px[..., :, :-1] - px[..., :, 1:]
    div[..., :, -1] += px[..., :, -1]
    div[..., 0, :] += -py[..., 0, :]
    div[..., 1:-1, :] += py[..., :-1, :] - py[..., 1:, :]
    div[..., -1, :] += py[..., -1, :]

    return div

@torch.no_grad()
def edge_weights_from_superpixels(spix, w_in=1.0, w_cross=0.1):

    spix = spix.long()
    same_x = (spix[..., :, 1:] == spix[..., :, :-1]).float()
    same_y = (spix[..., 1:, :] == spix[..., :-1, :]).float()
    wx = w_cross + (w_in - w_cross) * same_x
    wy = w_cross + (w_in - w_cross) * same_y
    return wx.unsqueeze(1), wy.unsqueeze(1)
def _lambertw(z):

    w = torch.log(z + 1.0)
    for _ in range(5):
        ew = torch.exp(w)
        wew = w * ew

        numerator = wew - z
        denominator = ew * (w + 1.0) + 1e-12

        w = w - numerator / denominator

    return w
def prox_kl_simplex(v, tau, a, b, n_bisect=30, eps=1e-12):

    if a <= 0:
        x = v + tau * b
        return project_simplex(x)

    c = tau * a
    d0 = v + tau * b - c

    with torch.no_grad():
        d0_max = d0.amax(dim=1, keepdim=True)
        d0_min = d0.amin(dim=1, keepdim=True)
        mu_low = d0_min - 20.0 * c
        mu_high = d0_max + 20.0 * c

    def q_of_mu(mu):
        z = (d0 - mu) / c
        z = z.clamp(min=-50.0, max=50.0)
        arg = torch.exp(z) / max(c, eps)
        w = _lambertw(arg)
        q = c * w.real
        return q.clamp_min(0.0)

    mu_l = mu_low
    mu_h = mu_high
    for _ in range(n_bisect):
        mu_m = 0.5 * (mu_l + mu_h)
        q_m = q_of_mu(mu_m)
        s = q_m.sum(dim=1, keepdim=True)
        mu_l = torch.where(s > 1.0, mu_m, mu_l)
        mu_h = torch.where(s > 1.0, mu_h, mu_m)

    q = q_of_mu(0.5 * (mu_l + mu_h))
    q = q / (q.sum(dim=1, keepdim=True).clamp_min(eps))
    return q


def project_simplex(x, eps=1e-12):

    B, K, H, W = x.shape
    x2 = x.permute(0, 2, 3, 1).reshape(-1, K)
    u, _ = torch.sort(x2, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - 1
    ind = torch.arange(1, K + 1, device=x.device).view(1, -1)
    cond = u - cssv / ind > 0
    rho = cond.sum(dim=1, keepdim=True).clamp_min(1)
    theta = cssv.gather(1, rho - 1) / rho
    q2 = (x2 - theta).clamp_min(0.0)
    q = q2.reshape(B, H, W, K).permute(0, 3, 1, 2)
    q = q / (q.sum(dim=1, keepdim=True).clamp_min(eps))
    return q

class ConvexFusionPrimalDual(nn.Module):

    def __init__(self, n_iters=80, tau=0.25, sigma=0.25, theta=1.0,
                 lam_c=1.0, lam_d=1.0, beta=0.2, eps=1e-12):
        super().__init__()
        self.n_iters = n_iters
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.lam_c = lam_c
        self.lam_d = lam_d
        self.beta = beta
        self.eps = eps

    def forward(self, pC, pD, wx=None, wy=None, q0=None):
        eps = self.eps
        pC = pC.clamp_min(eps);
        pC = pC / pC.sum(dim=1, keepdim=True)
        pD = pD.clamp_min(eps);
        pD = pD / pD.sum(dim=1, keepdim=True)
        B, K, H, W = pC.shape
        if wx is None: wx = torch.ones((B, 1, H, W - 1), device=pC.device, dtype=pC.dtype)
        if wy is None: wy = torch.ones((B, 1, H - 1, W), device=pC.device, dtype=pC.dtype)
        a = float(self.lam_c + self.lam_d)
        b = self.lam_c * torch.log(pC) + self.lam_d * torch.log(pD)

        if q0 is None:
            fused_log = b / max(a, eps)
            q = torch.softmax(fused_log, dim=1)
        else:
            q = q0

        q_bar = q.clone()
        px = torch.zeros((B, K, H, W - 1), device=pC.device, dtype=pC.dtype)
        py = torch.zeros((B, K, H - 1, W), device=pC.device, dtype=pC.dtype)

        for _ in range(self.n_iters):
            q_prev = q

            gx, gy = grad_forward(q_bar)
            px = px + self.sigma * gx
            py = py + self.sigma * gy

            lim_x = self.beta * wx
            lim_y = self.beta * wy
            px = torch.clamp(px, min=-lim_x, max=lim_x)
            py = torch.clamp(py, min=-lim_y, max=lim_y)

            div_p = div_backward(px, py)
            v = q - self.tau * div_p
            q = prox_kl_simplex(v, tau=self.tau, a=a, b=b, n_bisect=10, eps=eps)


            q_bar = q + self.theta * (q - q_prev)

        y = torch.argmax(q, dim=1)
        return q, y

def refine_label_convex_optimization(
        prob_map_1,
        prob_map_2,
        combined_path_str,
        dataset,
        cfg,
        lam_c: float = 1.0,
        lam_d: float = 0.2,
        beta: float = 0.10,
        n_iters: int = 100,
        eps: float = 1e-6
):

    img_path = os.path.join(dataset.data_root, combined_path_str + '.tif')
    fb_region_mask_numpy = felzenszwalb_img2labels(image_path=img_path)
    device = prob_map_1.device
    sp = torch.from_numpy(fb_region_mask_numpy).long().to(device).unsqueeze(0)

    def prep_prob(p):
        if p.min() < 0 or p.sum(dim=0).mean() > 1.5:
            p = F.softmax(p, dim=0)
        return p.clamp(min=eps, max=1.0).unsqueeze(0)

    P1 = prep_prob(prob_map_1)
    P2 = prep_prob(prob_map_2)

    wx, wy = edge_weights_from_superpixels(sp, w_in=1.0, w_cross=0.10)
    solver = ConvexFusionPrimalDual(
        n_iters=n_iters,
        lam_c=lam_c,
        lam_d=lam_d,
        beta=beta,
        tau=0.25, sigma=0.25, theta=1.0
    ).to(device)
    q_final, y_final = solver(P1, P2, wx=wx, wy=wy)
    corrected_label_indexed = y_final.squeeze(0).long()
    pseudo_label_for_loss = corrected_label_indexed.clone()
    return corrected_label_indexed, pseudo_label_for_loss, fb_region_mask_numpy