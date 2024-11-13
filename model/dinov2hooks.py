from .dinov2 import Dinov2
import torch
import torch.nn as nn

class Dinov2HOOKS:
    FACETS = ['attn','key', 'query', 'value', 'token']
    def __init__(self, version = "vit_giant2", vit_path_size=14, weights_dir=None):
        model = Dinov2(version)
        state_dict = model.load_model_weights(weights_dir)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        self.model = model
        self.hook_handlers = []
        self.vit_patch_size = vit_path_size
    
    def preprocess(self, prep_img):
        
        prep_img = self.make_input_divisible(prep_img)
        _w, _h = prep_img.shape[-2:]
        _h, _w = _h // self.vit_patch_size, _w // self.vit_patch_size
        
        return prep_img
    
    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size

        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def _get_hook(self, facet):
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers, facets):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if 'token' in facets:
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook('token')))
                if 'attn' in facets:
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook('attn')))
                for facet in ['key', 'query', 'value']:
                    if facet in facets:
                        self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch, layers = [39], facets = [FACETS[4]]):
        
        B, C, H, W = batch.shape
        self._feats = []

        self._register_hooks(layers, facets)
        _ = self.model(batch)
        self._unregister_hooks()

        return self._feats