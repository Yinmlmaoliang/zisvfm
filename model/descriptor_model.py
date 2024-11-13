import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dinov2hooks import Dinov2HOOKS

class DescriptorModel():
    def __init__(
        self,
        model_type="vit_small",
        vit_patch_size=14,
        dim=384,
        weights_dir="./weights"
        ):  
        # ----------------------
        # Encoder
        self.vit_version = model_type
        self.vit_encoder = Dinov2HOOKS(model_type, vit_patch_size, weights_dir)
        self.layers = [self.vit_encoder.model.depth-1] #index of layers we want to extract from
        self.vit_patch_size = vit_patch_size

        self.dim = dim
        self.register_tokens = self.vit_version in ["vit_giant2_reg", "vit_large_reg", "vit_base_reg", "vit_small_reg"]

    def preprocess(self, image):
        
        prep_img = self.make_input_divisible(image)
        _w, _h = prep_img.shape[-2:]
        _h, _w = _h // self.vit_patch_size, _w // self.vit_patch_size
        
        w_featmap = prep_img.shape[-2] // self.vit_patch_size
        h_featmap = prep_img.shape[-1] // self.vit_patch_size
            
        return prep_img, (w_featmap, h_featmap)

    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size

        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def extract_feats(self, image, facets):
        with torch.no_grad():
          outputs = self.vit_encoder._extract_features(image, self.layers, facets)
        attn, feats = outputs
        return attn, feats.transpose(1, 2).float()
    
    def compute_attention_entropy_weights(self, att: torch.Tensor) -> torch.Tensor:
        
        def compute_entropy(img):
            # Add a small constant to avoid log(0)
            epsilon = 1e-10
            img = img / img.sum()
            # Clamp values to avoid log(0) and normalize to prevent log values from exploding.
            return -torch.sum((img + epsilon) * torch.log2(img + epsilon))

        nb, nh, _, _ = att.shape
        entropies = torch.zeros(nb, nh).to(att.device)
        for b in range(nb):
            for h in range(nh):
                entropies[b, h] = compute_entropy(att[b, h])

        weights = torch.log(torch.sum(entropies, dim=1)[:, None] / (entropies))
        # weights = weights / weights.sum()
        return weights, entropies

    def get_background_patch_index(self, attentions: torch.Tensor, features: torch.Tensor, 
                              feature_map_dims: tuple, use_entropy_weights: bool = True) -> tuple:
        """
        Compute background patch index and cosine similarity using attention maps and features.
        
        Args:
            attentions: Attention maps tensor of shape (batch_size, num_heads, num_tokens, num_tokens)
            features: Feature tensor of shape (batch_size, num_tokens, height, width)
            feature_map_dims: Tuple of (width, height) for feature map dimensions
            use_entropy_weights: Whether to apply entropy-based weights to attention maps
        
        Returns:
            tuple: (background_patch_indices, weighted_cosine_similarity)
                - background_patch_indices: Indices of background patches for each image in batch
                - weighted_cosine_similarity: Cosine similarity matrix weighted by attention entropy
        """
        batch_size = attentions.shape[0]
        num_heads = attentions.shape[1]
        num_tokens = attentions.shape[2]
        width_feat, height_feat = feature_map_dims
        feature_dim = features.shape[3]
        
        # Reshape features for processing
        features = features.reshape(batch_size, num_tokens, -1)
        
        # Extract output patch attention
        token_offset = 5 if self.register_tokens else 1
        attention = attentions[:, :, 0, token_offset:].reshape(batch_size, num_heads, -1)
        attention = attention.reshape(batch_size, num_heads, width_feat, height_feat)

        # Compute attention entropy weights
        weights, _ = self.compute_attention_entropy_weights(attention)
        
        # Extract relevant feature descriptors
        descriptors = features[:, token_offset:, ]
        
        # Compute weighted and unweighted feature descriptors
        desc_shape = (batch_size, -1, num_heads, feature_dim)
        descriptors_unweighted = descriptors.reshape(desc_shape).reshape(batch_size, -1, num_heads * feature_dim)
        descriptors_weighted = (descriptors.reshape(desc_shape) * weights[:, None, :, None]).reshape(
            batch_size, -1, num_heads * feature_dim
        )
        
        # Normalize descriptors
        descriptors_unweighted = F.normalize(descriptors_unweighted, dim=-1, p=2)
        descriptors_weighted = F.normalize(descriptors_weighted, dim=-1, p=2)
        
        # Compute cosine similarities
        cosine_sim_weighted = torch.bmm(descriptors_weighted, descriptors_weighted.permute(0, 2, 1))
        
        # Apply entropy weights to attention if specified
        if use_entropy_weights:
            attention = attention * weights[:, :, None, None]
        
        # Find background patch indices
        background_indices = torch.argmin(torch.sum(attention, axis=1).reshape(batch_size, -1), dim=-1)
        
        # Reshape cosine similarity matrix
        cosine_sim_weighted = cosine_sim_weighted.reshape(batch_size, -1, width_feat * height_feat)
        
        return background_indices, cosine_sim_weighted