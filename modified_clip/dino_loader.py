import os
import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from .vision_transformer import vit_base


class DinoModel:
    def __init__(self, weights_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(weights_path)
        self.dino_input_size = (384, 384)
    def _load_model(self, weights_path: str) -> torch.nn.Module:
        print("[DINO (ViT-Base/8)] Model weights loaded...")
        model = vit_base(patch_size=8, num_classes=0)

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"no {weights_path} weights.")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

        checkpoint_key = "teacher"
        if checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def get_all_layer_outputs(self, input_tensor: torch.Tensor):

        input_tensor = input_tensor.to(self.device)
        if input_tensor.shape[-2:] != self.dino_input_size:
            input_tensor = torch.nn.functional.interpolate(
                input_tensor, size=self.dino_input_size, mode='bicubic', align_corners=False)
        list_of_layer_outputs, list_of_attentions = self.model.get_intermediate_layers(input_tensor, n=12)
        all_layer_features_stacked = torch.stack(list_of_layer_outputs, dim=0)
        last_layer_attention_map = list_of_attentions[-1]
        if input_tensor.shape[0] == 1:
            all_layer_features_stacked = all_layer_features_stacked.squeeze(1)
            last_layer_attention_map = last_layer_attention_map.squeeze(0)
        return all_layer_features_stacked, last_layer_attention_map

    @torch.no_grad()
    def build_knn_transition(self, feat: torch.Tensor, k: int = 30, sigma: float = 0.07):
        feat = F.normalize(feat, p=2, dim=-1)
        N = feat.shape[0]
        sim = feat @ feat.t()
        sim.fill_diagonal_(-1e9)
        vals, idx = torch.topk(sim, k=k, dim=1)
        w = torch.exp(vals / sigma)
        P = torch.zeros((N, N), device=feat.device, dtype=feat.dtype)
        P.scatter_(1, idx, w)
        P = P / (P.sum(dim=1, keepdim=True) + 1e-6)
        return P
