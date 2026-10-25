# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn.functional as F
from torch import nn


class Dinov3Backbone(nn.Module):
    def __init__(
        self, name="dinov2_vitb14", pretrained_weight=None, cfg=None, *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg

        import os

        local_repo = os.environ.get("DINOV3_REPO_DIR", "")
        if local_repo and os.path.isdir(local_repo):
            self.encoder = torch.hub.load(
                local_repo,
                self.name,
                source="local",
                pretrained=False,
                drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
            )
        else:
            self.encoder = torch.hub.load(
                "facebookresearch/dinov3",
                self.name,
                source="github",
                pretrained=False,
                drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
            )
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim
        self._install_rope_cache()

    def _install_rope_cache(self):
        rope_embed = getattr(self.encoder, "rope_embed", None)
        if rope_embed is None or hasattr(rope_embed, "_rope_cache"):
            return
        rope_embed._rope_cache = {}
        rope_pad_cache = {}
        rope_embed._rope_pad_cache = rope_pad_cache
        orig_forward = rope_embed.forward

        def cached_forward(*, H: int, W: int):
            if rope_embed.training:
                return orig_forward(H=H, W=W)
            periods = getattr(rope_embed, "periods", None)
            dev = periods.device if periods is not None else None
            dt = getattr(rope_embed, "dtype", None)
            key = (H, W, dev, dt)
            cached = rope_embed._rope_cache.get(key)
            if cached is None:
                cached = orig_forward(H=H, W=W)
                rope_embed._rope_cache[key] = cached
            return cached

        rope_embed.forward = cached_forward

        for blk in getattr(self.encoder, "blocks", []):
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            qkv_mod = getattr(attn, "qkv", None)
            if qkv_mod is not None and hasattr(qkv_mod, "bias_mask"):
                orig_qkv_forward = qkv_mod.forward

                def make_fast_qkv_forward(m, fallback):
                    cache = {}

                    def fast_qkv_forward(inp):
                        if m.training or m.bias is None:
                            return fallback(inp)
                        k = (m.bias.device, m.bias.dtype)
                        mb = cache.get(k)
                        if mb is None:
                            mb = m.bias * m.bias_mask.to(m.bias.dtype)
                            cache[k] = mb
                        return F.linear(inp, m.weight, mb)

                    return fast_qkv_forward

                qkv_mod.forward = make_fast_qkv_forward(qkv_mod, orig_qkv_forward)

            orig_compute_attn = attn.compute_attention

            def make_fast_compute_attn(a, fallback):
                num_heads = a.num_heads
                C = a.qkv.in_features
                head_dim = C // num_heads
                half_dim = head_dim // 2

                def fast_compute_attn(qkv, attn_bias=None, rope=None):
                    if a.training or attn_bias is not None or rope is None:
                        return fallback(qkv, attn_bias=attn_bias, rope=rope)
                    sin, cos = rope
                    if sin.ndim != 2:
                        return fallback(qkv, attn_bias=attn_bias, rope=rope)
                    B, N, _ = qkv.shape
                    pad_key = (N, sin.shape[0], sin.device, qkv.dtype)
                    padded = rope_pad_cache.get(pad_key)
                    if padded is None:
                        prefix = N - sin.shape[0]
                        sin_c = sin[:, :half_dim].to(dtype=qkv.dtype)
                        cos_c = cos[:, :half_dim].to(dtype=qkv.dtype)
                        sin_p = (
                            F.pad(sin_c, (0, 0, prefix, 0), value=0.0)
                            .view(1, 1, 1, N, half_dim)
                            .contiguous()
                        )
                        cos_p = (
                            F.pad(cos_c, (0, 0, prefix, 0), value=1.0)
                            .view(1, 1, 1, N, half_dim)
                            .contiguous()
                        )
                        padded = (sin_p, cos_p)
                        rope_pad_cache[pad_key] = padded
                    sin_pad, cos_pad = padded
                    qkv_p = (
                        qkv.view(B, N, 3, num_heads, head_dim)
                        .permute(0, 2, 3, 1, 4)
                        .contiguous()
                    )
                    qk = qkv_p[:, :2]
                    v = qkv_p[:, 2]
                    qk1 = qk[..., :half_dim]
                    qk2 = qk[..., half_dim:]
                    qk_rot = torch.cat(
                        [qk1 * cos_pad - qk2 * sin_pad, qk2 * cos_pad + qk1 * sin_pad],
                        dim=-1,
                    )
                    out = F.scaled_dot_product_attention(qk_rot[:, 0], qk_rot[:, 1], v)
                    return out.transpose(1, 2).reshape(B, N, C)

                return fast_compute_attn

            attn.compute_attention = make_fast_compute_attn(attn, orig_compute_attn)

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"

        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

        return y

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.encoder.n_blocks + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens"):
            layer_depth = 0
        elif param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
