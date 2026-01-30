from functools import reduce
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from modified_clip.model import CLIP



class Pipeline(nn.Module):
    def __init__(self, cfg, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.clip, self.attn_refine = None, None
        self.device = device
        if cfg.model_name in ["ViT-B/16", "ViT-L/14"]:
            self.clip = CLIP(cfg, model_name=cfg.model_name, logit_scale=cfg.logit_scale, attn_type=cfg.attn_type,
                                 fuse_feature=cfg.fuse_feature, size=cfg.size, device=self.device)
        else:
            raise NotImplementedError("Unknown Model")
        self.cfg = cfg
    def diffuse_scores(self,P: torch.Tensor, S0: torch.Tensor, steps: int = 40, alpha: float = 0.9):

        S = S0
        for _ in range(steps):
            S = alpha * (P @ S) + (1 - alpha) * S0
        return S
    def forward(self, ori_img, img, classify_fg_text_features, classify_bg_text_features):

        segment_results = self.clip(img, classify_fg_text_features, classify_bg_text_features, ori_img)
        seg = segment_results["seg"]
        transition_matrix_dino = segment_results["transition_matrix_dino"]
        seg_matched = segment_results["seg_matched"]
        seg_matched_score = seg_matched.amax(dim=0)
        seg_trans = self.diffuse_scores(transition_matrix_dino.half(), seg_matched, steps=40, alpha=0.9)
        seg_trans_map = seg_trans.transpose(0, 1).reshape(seg_trans.shape[1], int(seg_trans.shape[0] ** 0.5), int(seg_trans.shape[0] ** 0.5)).float()
        transition_matrix_clip = segment_results["transition_matrix_clip"]
        dino_seg_matched = segment_results["dino_seg_matched"]
        dino_matched_score = dino_seg_matched.amax(dim=0)
        dino_trans = self.diffuse_scores(transition_matrix_clip.half(), dino_seg_matched.half(), steps=40, alpha=0.9)
        dino_trans_map = dino_trans.transpose(0, 1).reshape(dino_trans.shape[1], self.clip.img_h, self.clip.img_w).float()

        return seg_trans_map, seg_matched_score, dino_trans_map, dino_matched_score