import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import os


class DinoV3Model:


    def __init__(self,
                 model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                 device: str = 'cuda',
                 input_size=(512, 512)):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dino_input_size = input_size
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config.output_attentions = True
        self.num_register_tokens = getattr(self.config, "num_register_tokens", 0)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=True,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.final_norm = self.model.norm
        print(f"[DINOv3 (ViT-Base/16)] Model weights loaded...")

    @torch.no_grad()
    def get_all_layer_outputs(self, input_tensor: torch.Tensor):

        input_tensor = input_tensor.to(self.device)

        if input_tensor.shape[-2:] != self.dino_input_size:
            input_tensor = F.interpolate(
                input_tensor,
                size=self.dino_input_size,
                mode='bicubic',
                align_corners=False
            )

        outputs = self.model(
            pixel_values=input_tensor,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        raw_hidden_states = outputs.hidden_states[1:]
        last_attn_raw = outputs.attentions[-1] if outputs.attentions else None
        processed_layers = []
        keep_indices = [0] + list(range(1 + self.num_register_tokens, raw_hidden_states[0].shape[1]))
        start_patch_idx = 1 + self.num_register_tokens
        for state in raw_hidden_states:
            if self.final_norm:
                state = self.final_norm(state)
            if self.num_register_tokens > 0:
                cls_token = state[:, 0:1, :]
                patch_tokens = state[:, start_patch_idx:, :]
                state = torch.cat([cls_token, patch_tokens], dim=1)

            processed_layers.append(state)
        all_layer_features_stacked = torch.stack(processed_layers, dim=0)

        last_layer_attention_map = None
        if last_attn_raw is not None:

            if self.num_register_tokens > 0:
                idx_tensor = torch.tensor(keep_indices, device=self.device)
                last_layer_attention_map = last_attn_raw.index_select(-2, idx_tensor).index_select(-1, idx_tensor)
            else:
                last_layer_attention_map = last_attn_raw

        if input_tensor.shape[0] == 1:
            all_layer_features_stacked = all_layer_features_stacked.squeeze(1)
            if last_layer_attention_map is not None:
                last_layer_attention_map = last_layer_attention_map.squeeze(0)
        return all_layer_features_stacked, last_layer_attention_map

def main():
    extractor = DinoV3Model(
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        device="cuda",
        input_size=(512, 512),
    )
    x = torch.randn(1, 3, 512, 512)
    all_feats, last_attn = extractor.get_all_layer_outputs(x)
    print("all_feats.shape =", tuple(all_feats.shape))
    if last_attn is None:
        print("last_attn = None")
    else:
        print("last_attn.shape =", tuple(last_attn.shape))
    if all_feats.dim() == 3:
        L, T, C = all_feats.shape
        patch_tokens = T - 1
        g = int(patch_tokens ** 0.5)
        print(f"L={L}, T={T}, C={C}, patch_tokens={patch_tokens}, patch_grid≈{g}x{g}")

if __name__ == "__main__":
    main()