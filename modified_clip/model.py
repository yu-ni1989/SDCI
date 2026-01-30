import math

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as ttf
from PIL import Image
import os

from modified_clip.dino_loader import DinoModel
from modified_clip.dinov2 import DinoV2Model
from modified_clip.dinov3 import DinoV3Model

class CLIP(nn.Module):
    def __init__(self, cfg, model_name="ViT-L/14", attn_type="fused-attn", fuse_feature=True,
                 logit_scale=100, device="cuda", size=512,
                 ):
        super(CLIP, self).__init__()

        self.version = cfg.dino_version
        self.device = device
        self.attn_type = attn_type
        self.fuse_feature = fuse_feature
        self.size = size

        model, preprocess = clip.load(model_name, device=device)
        self.model = model.eval()

        self.preprocess = ttf.Compose([self._resize] + preprocess.transforms[2:])
        self.patch_size = int(model_name.split("/")[-1])
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale), requires_grad=False)

        self.layers = model.visual.transformer.layers
        self.modify()
        self.model.eval()
        self.img_h, self.img_w = None, None
        self.attn = None
        self.img_part_features = None
        self.image_feature = []
        self.dino_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.dino_model = DinoModel(
             weights_path="C:/Users/UU/Desktop/SDCI/dino_vitbase8_pretrain.pth",
             device=self.device)
        self.dino_v2_model = DinoV2Model(
            model_name="dinov2_vitb14_reg",
            device=self.device)
        self.dino_v3_model = DinoV3Model(
            model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
            device=self.device)

    def modify(self):
        model_transformer = self.model.visual.transformer
        model_visual = self.model.visual

        def custom_attn(attn_layer, x, attn_mask=None):

            num_heads = attn_layer.num_heads
            _, bsz, embed_dim = x.size()
            head_dim = embed_dim // num_heads
            scale = head_dim ** -0.5

            q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
            q, k, v = q.clone(), k.clone(), v.clone()
            q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

            if self.attn_type == "fused-attn" and attn_mask is not None:
                attn_mask /= torch.sum(attn_mask, dim=-2, keepdim=True)
                attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
                attn_mask = (attn_mask + attn_mask.transpose(-2, -1)) / 2
                attn_mask -= attn_mask.mean(-2, keepdim=True)
                attn_mask = torch.clamp(attn_mask, 0)
                attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
                attn_mask = attn_mask.flatten(0, 1)
                attn_weights = torch.repeat_interleave(attn_mask, dim=0, repeats=v.shape[0] // attn_mask.shape[0])
            elif self.attn_type == "q-q":
                attn_weights = torch.bmm(q * scale, q.transpose(1, 2))
                attn_weights = F.softmax(attn_weights, dim=-1)
            elif self.attn_type == "k-k":
                attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
                attn_weights = F.softmax(attn_weights, dim=-1)
            elif self.attn_type == "v-v":
                attn_weights = torch.bmm(v * scale, v.transpose(1, 2))
                attn_weights = F.softmax(attn_weights, dim=-1)
            elif self.attn_type == "vanilla":
                attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
                attn_weights = F.softmax(attn_weights, dim=-1)
            else:
                identity = torch.eye(v.shape[-2], dtype=v.dtype, device=v.device)[None]
                attn_weights = torch.repeat_interleave(identity, dim=0, repeats=v.shape[0])

            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
            attn_output = attn_layer.out_proj(attn_output)

            return attn_output, attn_weights

        def forward(x: torch.Tensor):
            h, w = x.shape[-2], x.shape[-1]
            positional_embedding_new = self.upsample_pos_emb(model_visual.positional_embedding,
                                                             (h // self.patch_size, w // self.patch_size))
            x = model_visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([model_visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                  dtype=x.dtype, device=x.device), x],
                          dim=1)
            x = x + positional_embedding_new.to(x.dtype)
            x = model_visual.ln_pre(x)

            x = x.permute(1, 0, 2)

            return model_visual.transformer(x)

        def forward_transformer(x: torch.Tensor, attn_mask: torch.Tensor = None):
            attn_maps = 0
            intermediate_features = []
            for i in range(self.layers - 1):
                ln_x = model_transformer.resblocks[i].ln_1(x)
                if self.fuse_feature:
                    intermediate_features.append(ln_x)
                ln_x, attn_map = model_transformer.resblocks[i].attn(ln_x, ln_x, ln_x, need_weights=True,
                                                                     attn_mask=attn_mask, average_attn_weights=False)
                attn_maps += attn_map
                x = x + ln_x
                x = x + model_transformer.resblocks[i].mlp(model_transformer.resblocks[i].ln_2(x))
            model_res = model_transformer.resblocks[-1]
            intermediate_features.append(x)
            processed_features = []
            last_attn = None
            for x_feat in intermediate_features:
                ln_x, attn = custom_attn(model_res.attn, model_res.ln_1(x_feat), attn_mask=attn_maps)
                processed_features.append(ln_x)
                last_attn = attn
            img_features = torch.stack(processed_features)
            img_features = model_visual.ln_post(img_features.squeeze())
            if model_visual.proj is not None:
                img_features = img_features @ model_visual.proj

            return img_features, last_attn

        model_transformer.forward = forward_transformer
        model_visual.forward = forward

    def classify(self, x: torch.Tensor, text_emb: torch.Tensor):
        x = x / x.norm(dim=-1, keepdim=True)
        norm_text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_per_image = self.logit_scale * x @ norm_text_emb.to(x.dtype).t()

        soft_per_image = logit_per_image.softmax(dim=-1)
        return soft_per_image, logit_per_image

    def _resize(self, image):
        ori_width, ori_height = image.size
        ratio = self.size / min(ori_width, ori_height)
        ori_width, ori_height = ori_width * ratio, ori_height * ratio
        h, w = (int(ori_height / self.patch_size + 0.5) * self.patch_size,
                int(ori_width / self.patch_size + 0.5) * self.patch_size)
        resized_image = image.resize((w, h), Image.BICUBIC)
        return resized_image

    @staticmethod
    def upsample_pos_emb(emb, new_size):
        first, emb = emb[:1, :], emb[1:, :]
        n, d = emb.size(0), emb.size(1)
        size = int(np.sqrt(n))
        emb = emb.permute(1, 0).view(1, d, size, size)
        emb = F.interpolate(emb, size=new_size, mode='bilinear')
        emb = emb.view(d, -1).contiguous().permute(1, 0)
        emb = torch.cat([first, emb], 0)
        return emb.half()

    def classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                classname_splits = []
                if ',' in classname:
                    classname_splits = [s.strip() for s in classname.split(',')]
                    texts = []
                    for template in templates:
                        for cls_split in classname_splits:
                            texts.append(template.format(cls_split))
                else:
                    texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).to(self.device)
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

                if classname_splits:
                    class_embeddings = class_embeddings.reshape(len(templates), len(classname_splits), -1)
                    class_embeddings = class_embeddings.mean(dim=1)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights.t()

    @staticmethod
    def resample1(tensor, size):
        return F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(size, size), mode='bilinear',
                             align_corners=False).squeeze()
    @staticmethod
    def resample(tensor, target_len):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        tensor = tensor.float()
        N_in = tensor.shape[0]
        size_in = int(math.sqrt(N_in))
        size_out = int(math.sqrt(target_len))
        tensor = tensor.view(1, N_in, size_in, size_in)
        tensor = F.interpolate(tensor, size=(size_out, size_out), mode='bilinear', align_corners=False)
        tensor = tensor.view(N_in, target_len)
        tensor = tensor.t()
        tensor = tensor.view(1, target_len, size_in, size_in)
        tensor = F.interpolate(tensor, size=(size_out, size_out), mode='bilinear', align_corners=False)
        tensor = tensor.view(target_len, target_len)
        tensor = tensor.t()
        return tensor
    @staticmethod
    def normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        denominator = max_val - min_val
        return (tensor - min_val) / denominator

    @staticmethod
    def upsample_seq(seq, target_N):
        N_in, C = seq.shape
        size_in = int(math.sqrt(N_in))
        size_out = int(math.sqrt(target_N))
        img = seq.t().reshape(1, C, size_in, size_in)
        out_img = F.interpolate(
            img,
            size=(size_out, size_out),
            mode='bilinear',
            align_corners=False
        )
        out_seq = out_img.permute(0, 2, 3, 1).reshape(-1, C)
        return out_seq

    def min_max_normalize(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    def forward(self, img: torch.Tensor, fg_text_features: torch.Tensor, bg_text_features: torch.Tensor, ori_img):
        self.img_h, self.img_w = img.shape[2] // self.patch_size, img.shape[3] // self.patch_size
        text_features = torch.cat([fg_text_features, bg_text_features, fg_text_features.mean(0, True)], dim=0)
        with torch.no_grad():
            img_feature, attn = self.model.encode_image(img)
            imgs_norm = self.dino_transform(ori_img)
            if self.version == 1:
                dino_all_feature_maps, dino_last_layer_attention = self.dino_model.get_all_layer_outputs(imgs_norm)
            elif self.version == 2:
                dino_all_feature_maps, dino_last_layer_attention = self.dino_v2_model.get_all_layer_outputs(imgs_norm)
            else:
                raise NotImplementedError("Unknown dino_version")

            transition_matrix_dino = self.dino_model.build_knn_transition(dino_all_feature_maps[-1, 1:])
            transition_matrix_clip = self.dino_model.build_knn_transition(img_feature[-1, 1:].float())
            dino_to_clip_attention = self.resample(dino_last_layer_attention.mean(0)[1:, 1:], attn.shape[1] - 1)
            clip_to_dino_attention = self.resample(attn.mean(0)[1:, 1:], dino_last_layer_attention.shape[1] - 1)
            dino_all_feature_maps = self.classify(dino_all_feature_maps, text_features)[0][:, 1:, :len(fg_text_features)]
            dino_last = dino_all_feature_maps[-1]
            dino_last[dino_last < dino_last.amax(0, keepdim=True) * 0.2] = 0
            seg = self.classify(img_feature, text_features)[0][:, 1:, :len(fg_text_features)]
            seg_last = seg[-1]
            seg_last[seg_last < seg_last.amax(0, keepdim=True) * 0.2] = 0
            dino_seg = (dino_last_layer_attention.mean(0)[1:, 1:] + clip_to_dino_attention) @ dino_last + dino_all_feature_maps[:-1].mean(0)
            seg = (attn.mean(0)[1:, 1:] + dino_to_clip_attention.half()) @ seg_last + seg[:-1].mean(0)
            seg_matched = self.upsample_seq(seg, target_N=transition_matrix_dino.shape[0])
            dino_seg_matched = self.upsample_seq(dino_seg, target_N=transition_matrix_clip.shape[0])

            return {"seg": seg.detach(), "img_part_features": img_feature.clone(),
                    "mid_feature": None, "attn_map": attn.mean(0)[1:, 1:].clone(),
                    "dino_seg": dino_seg.detach(),
                    "transition_matrix_dino": transition_matrix_dino.detach(), "seg_matched": seg_matched.detach(),
                    "transition_matrix_clip": transition_matrix_clip.detach(), "dino_seg_matched": dino_seg_matched.detach()}

