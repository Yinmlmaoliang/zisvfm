import torch
from torch import nn
import os
import math
from functools import partial
from .transformer_layers.patch_embed import PatchEmbedding
from .transformer_layers.swiglu_ffn import SwiGLUFFN
from .transformer_layers.Transformer import Transformer
from .transformer_layers.mlp import Mlp
"""
Inspiration from https://github.com/facebookresearch/dinov2
Simplified EVAL version of the DINO v2 model,
model weights are under facebookresearch/dinov2 CCC LICENSE, read before using
"""
CONFIGS = {
    "vit_giant2": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_layer": SwiGLUFFN,
    },
    "vit_large": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_layer": Mlp,
    },
    "vit_base": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_layer": Mlp,
    },
    "vit_small": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_layer": Mlp,
    },
    "vit_giant2_reg": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_layer": SwiGLUFFN,
        "num_register_tokens":4
    },
    "vit_large_reg": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_layer": Mlp,
        "num_register_tokens":4
    },
    "vit_base_reg": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_layer": Mlp,
        "num_register_tokens":4,
    },
    "vit_small_reg": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_layer": Mlp,
        "num_register_tokens":4,
}}
CHECKPOINTS = {"vit_giant2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
               "vit_large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
               "vit_base": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
               "vit_small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
               "vit_giant2_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
               "vit_large_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
               "vit_base_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
               "vit_small_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"}

class Dinov2(nn.Module):
    def __init__(self, version = "vit_giant2"): 
        super().__init__()
        assert version in CONFIGS.keys(), f"version {version} not in {list(CONFIGS.keys())}"
        args = CONFIGS[version]

        self.model_version = version
        self.depth = args["depth"]
        self.patch_size = args["patch_size"]
        self.need_register_tokens = 'num_register_tokens' in args

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(img_size= args["img_size"], patch_size= args["patch_size"], embed_dim= args["embed_dim"])
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args["embed_dim"]))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, args["embed_dim"]))
        blocks_list = [
            Transformer(
                dim= args["embed_dim"],
                num_heads=args["num_heads"],
                norm_layer=norm_layer,
                ffn_layer= args["ffn_layer"],
            )
            for _ in range(self.depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(args["embed_dim"])
        if self.need_register_tokens:
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, args['num_register_tokens'], args['embed_dim'])) if args['num_register_tokens'] else None
            )
    
    def load_model_weights(self, weights_dir="./weights"):
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        
        weight_file_path = os.path.join(weights_dir, f"{self.model_version}.pth")
        
        if not os.path.exists(weight_file_path):
            print(f"Downloading weights for {self.model_version}...")
            checkpoint_url = CHECKPOINTS[self.model_version]
            model_weights = torch.hub.load_state_dict_from_url(
                checkpoint_url, 
                weights_dir, 
                map_location="cpu",
                file_name=f"{self.model_version}.pth"
            )
        else:
            model_weights = torch.load(weight_file_path, map_location="cpu")
        
        return model_weights

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1 #remove cls token
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self,x):
        B, nc, w, h = x.shape #B C H W
        x = self.patch_embed(x) #B N D
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) #B N+1 D
        x = x + self.interpolate_pos_encoding(x, w, h)

        if  self.need_register_tokens:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)    
        return x
    
    def get_last_selfattention(self, x):
        B, nc, w, h = x.shape #B C H W
        x = self.patch_embed(x) #B N D
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) #B N+1 D
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
            # return attention of the last block
              return blk(x, return_attention=True)