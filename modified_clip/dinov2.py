import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class DinoV2Model:
    def __init__(self, model_name: str = "dinov2_vitl14_reg", device: str = "cuda", input_size=(532, 532),
                 hub_dir=None):
        warnings.filterwarnings("ignore", message="xFormers is not available")
        self.device = torch.device(device)
        self.model_name = model_name
        self.dino_input_size = input_size
        self.hub_dir = hub_dir
        self.num_register_tokens = 4 if "_reg" in model_name else 0
        self.model = self._load_model()
    def _load_model(self) -> torch.nn.Module:
        if self.hub_dir is not None:
            torch.hub.set_dir(self.hub_dir)
        print(f"[DINOv2({self.model_name})] Model weights loaded...")
        model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        model.to(self.device).eval()
        return model
    def _flatten_blocks_in_order(self):
        # 处理 DINOv2 可能存在的 chunked blocks 结构
        if getattr(self.model, "chunked_blocks", False):
            blocks = []
            for chunk in self.model.blocks:
                for b in chunk:
                    if not isinstance(b, nn.Identity):
                        blocks.append(b)
            return blocks
        else:
            return list(self.model.blocks)
    @torch.no_grad()
    def get_all_layer_outputs(self, input_tensor: torch.Tensor, return_spatial_only=True):
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if self.dino_input_size is not None:
            h, w = self.dino_input_size
            if input_tensor.shape[-2:] != (h, w):
                input_tensor = F.interpolate(input_tensor, size=(h, w), mode="bicubic", align_corners=False)
        x = self.model.prepare_tokens_with_masks(input_tensor)
        blocks = self._flatten_blocks_in_order()
        outputs = []
        last_attn = None
        for i, blk in enumerate(blocks):
            if i == len(blocks) - 1:
                x_norm1 = blk.norm1(x)
                B, T, C = x_norm1.shape
                nh = blk.attn.num_heads
                head_dim = C // nh
                qkv = blk.attn.qkv(x_norm1).reshape(B, T, 3, nh, head_dim).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                q = q * blk.attn.scale
                last_attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            x = blk(x)
            x_normalized = self.model.norm(x)
            outputs.append(x_normalized)
        all_layer_features_stacked = torch.stack(outputs, dim=0)  # (L, B, T, C)
        if return_spatial_only:
            start_idx = self.num_register_tokens
            all_layer_features_stacked = all_layer_features_stacked[:, :, start_idx:, :]
        if input_tensor.shape[0] == 1:
            all_layer_features_stacked = all_layer_features_stacked.squeeze(1)
            if last_attn is not None:
                last_attn = last_attn.squeeze(0)

        return all_layer_features_stacked, last_attn[:, start_idx:, start_idx:]
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DinoV2Model(
        model_name="dinov2_vitb14_reg",
        device=device,
        input_size=(532, 532),
    )
    img_path = r""
    img = Image.open(img_path).convert("RGB")

    tfm = transforms.Compose([
        transforms.Resize((532, 532), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    x = tfm(img).unsqueeze(0).to(device)
    all_layers, last_attn = extractor.get_all_layer_outputs(x,return_spatial_only=True)
    print("all_layers shape =", tuple(all_layers.shape))
    print("last_attn shape  =", tuple(last_attn.shape))
    num_reg = getattr(extractor.model, "num_register_tokens", 0)
    last = all_layers[-1]
    patch_tok = last[1:]
    all_layers1 = all_layers[:, 1:, :]
    print("patch_tok shape  =", tuple(patch_tok.shape))
    print("all_layers1 shape  =", tuple(all_layers1.shape))
    save_path = r''
    torch.save(all_layers1, save_path)
    print(f"Saved patch tokens to {save_path}")

if __name__ == "__main__":
    main()