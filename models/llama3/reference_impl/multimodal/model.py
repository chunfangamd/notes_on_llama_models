# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fairscale.nn.model_parallel.initialize as fs_init

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from PIL import Image as PIL_Image

from torch import nn, Tensor
from torch.distributed import _functional_collectives as funcol

from ..model import apply_rotary_emb, ModelArgs, precompute_freqs_cis, RMSNorm

from .encoder_utils import (
    build_encoder_attention_mask,
    contract_num_tokens_from_mult8,
    expand_num_tokens_to_mult8,
    initialize_global_position_embedding_from_local,
    resize_global_position_embedding,
    resize_local_position_embedding,
)
from .image_transform import VariableSizeImageTransform
from .utils import get_negative_inf_value, to_2tuple


logger = logging.getLogger(__name__)
MP_SCALE = 8


def reduce_from_tensor_model_parallel_region(input_):
    """All-reduce the input tensor across model parallel group."""
    output = funcol.all_reduce(input_, "sum", group=fs_init.get_model_parallel_group())
    output = funcol.wait_tensor(output)
    return output


def gather_from_tensor_model_parallel_region(input_):
    """Gather tensors and concatenate along the last dimension."""

    world_size = fs_init.get_model_parallel_world_size()
    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = fs_init.get_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    output = funcol.all_gather_tensor(
        input_,
        gather_dim=last_dim,
        group=fs_init.get_model_parallel_group(),
    )
    output = funcol.wait_tensor(output)
    return output


def _get_full_row_masked_out_mask(
    attn_bias,
    negative_inf_value,
):
    """
    attn_bias should be a 4D tensor of shape [B, H, S1, S2]
    where B is the batch size, H is the number of heads,
    and S1/S2 are the sequence lengths. This returns
    a 4D tensor of shape [B, H, S1, 1] which stores boolean
    values which are 0 if the a full row in the last dimension
    contains negative infinity values, otherwise it's 1.
    Example:
        attn_bias: shape=(bsz, 1, ntext, nimg*nchunks*vision_seqlen)=(1, 1, 512, 2*4*1601)
        negative_inf_value: -inf
    """
    return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]


# Image encoder for inference
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        bias: Optional[bool] = False,
    ) -> None:
        """
        Example: 
            in_channels=3, output_channels=1280, kernel_size=14, stride=14
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)  # E.g., kernel_size=14, stride=14
        self._linear = ColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )  # (out_channels_per_device, in_channels*kernel_size[0]*kernel_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape=(num_stacked_chunks, nch, w, h). 
            Here, num_stacked_chunks = bsz*num_concurrent_media*max_num_chunks.
        E.g., x.shape=(8, 3, 560, 560)
        """
        # x: (num_stacked_chunks, in_channels, width, height) = (8, 3, 560, 560)
        # Lh = (H - Kh) // Sh + 1 = (560 - 14) // 14 + 1 = 40
        # Lw = (W - Kw) // Sw + 1 = (560 - 14) // 14 + 1 = 40
        # num_vision_tokens = Lh * Lw = 1600
        x = self._unfold(x)     # x: (num_stacked_chunks, in_channels*kernel_size[0]*kernel_size[1], num_vision_tokens) = (8, 588, 1600)
        x = x.permute(0, 2, 1)  # x: (num_stacked_chunks, num_vision_tokens, in_channels*kernel_size[0]*kernel_size[1]) = (8, 1600, 588)
        # x                    : (num_stacked_chunks, num_vision_tokens, in_channels*kernel_size[0]*kernel_size[1]) = (8, 1600, 588)
        # self._linear.weight^T: (in_channels*kernel_size[0]*kernel_size[1], out_channels_per_device) = (588, 1280/model_parallel_size)
        x = F.linear(x, self._linear.weight)             # x: (num_stacked_chunks, num_vision_tokens, out_channels_per_device) = (8, 1600, 1280/model_parallel_size)
        x = gather_from_tensor_model_parallel_region(x)  # x: (num_stacked_chunks, num_vision_tokens, out_channels) = (8, 1600, 1280)
        return x


class ImageFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        # layers
        self.c_fc = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.c_proj = RowParallelLinear(
            hidden_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x):
        hidden = F.linear(x, self.c_fc.weight, self.c_fc.bias)
        hidden = self.non_linearity(hidden)
        hidden = F.linear(hidden, self.c_proj.weight)
        hidden = reduce_from_tensor_model_parallel_region(hidden)
        hidden += self.c_proj.bias
        return hidden


class ImageAttention(nn.Module):
    def __init__(
        self,
        dim,       # 1280
        head_dim,  # 80
        n_heads,   # 16
    ):
        super().__init__()
        model_parallel_size = fs_init.get_model_parallel_world_size()
        qkvo_replication = 1
        if model_parallel_size > 16:
            qkvo_replication = model_parallel_size // 8

        self.n_kv_heads = n_heads
        self.n_local_heads = n_heads * qkvo_replication // model_parallel_size
        self.n_local_kv_heads = (
            self.n_kv_heads * qkvo_replication // model_parallel_size
        )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = ColumnParallelLinear(
            dim,
            qkvo_replication * n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # (qkvo_replication*n_local_heads*head_dim, dim)
        self.wk = ColumnParallelLinear(
            dim,
            qkvo_replication * self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # (qkvo_replication*n_local_kv_heads*head_dim, dim)
        self.wv = ColumnParallelLinear(
            dim,
            qkvo_replication * self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # (qkvo_replication*n_local_kv_heads*head_dim, dim)
        self.wo = RowParallelLinear(
            qkvo_replication * n_heads * self.head_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )  # (dim, qkvo_replication*n_local_heads*head_dim)
        self.qkvo_replication = qkvo_replication

    def forward(
        self,
        x: torch.Tensor,  # (bs, slen, dim)
        mask: torch.Tensor = None,
    ):  # --> (bs, slen, slen)
        """
        Example:
            x.shape=(2, 4*1608, 1280) : (bs, slen, dim)=(bsz*num_concurrent_media, num_chunks*ntks, dim)
            mask.shape=(2, 1, 4*1608, 4*1608)
        """

        xq, xk, xv = [
            F.linear(x, w) for w in [self.wq.weight, self.wk.weight, self.wv.weight]
        ]
        # x: (bs, slen, dim), wq: (qkvo_replication*n_local_heads*head_dim, dim) -> xq: (bs, slen, qkvo_replication*n_local_heads*head_dim)
        # x: (bs, slen, dim), wk: (qkvo_replication*n_local_kv_heads*head_dim, dim) -> xk: (bs, slen, qkvo_replication*n_local_kv_heads*head_dim)
        # x: (bs, slen, dim), wv: (qkvo_replication*n_local_kv_heads*head_dim, dim) -> xv: (bs, slen, qkvo_replication*n_local_kv_heads*head_dim)

        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)            # xq: (bs, slen, n_local_heads,    head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)  # xk: (bs, slen, n_local_kv_heads, head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)  # xv: (bs, slen, n_local_kv_heads, head_dim)

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]
        # xq: (bs, n_local_heads,    slen, head_dim)
        # xk: (bs, n_local_kv_heads, slen, head_dim)
        # xv: (bs, n_local_kv_heads, slen, head_dim)

        xk = xk.repeat_interleave(self.n_rep, dim=1)  # xk: (bs, n_heads, slen, head_dim)
        xv = xv.repeat_interleave(self.n_rep, dim=1)  # xv: (bs, n_heads, slen, head_dim)

        # attn_weight: xq @ xk^T / sqrt(head_dim) : (bs, n_heads, slen, head_dim) @ (bs, n_heads, head_dim, slen) -> (bs, n_heads, slen, slen)
        # attn_score: softmax(attn_weight + mask) : (bs, n_heads, slen, slen)
        # attn_output: attn_score @ xv : (bs, n_heads, slen, slen) @ (bs, n_heads, slen, head_dim) -> (bs, n_heads, slen, head_dim)
        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )  # attn_output: (bs, n_heads, slen, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)  # attn_output: (bs, slen, n_heads*head_dim)

        out = F.linear(attn_output, self.wo.weight) # attn_output: (bs, slen, n_heads*head_dim), self.wo.weight^T: (n_heads*head_dim, dim) -> out: (bs, slen, dim)
        out = reduce_from_tensor_model_parallel_region(out)  # out: (bs, slen, dim)
        out = out / self.qkvo_replication                    # out: (bs, slen, dim)
        return out


class ImageTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,  # 1280
        n_head: int,   # 16
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head                    # 16
        self.head_dim = d_model // self.n_heads  # 80
        self.attn = ImageAttention(
            dim=d_model,             # 1280
            head_dim=self.head_dim,  # 80
            n_heads=self.n_heads,    # 16
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = ImageFeedForward(
            dim=d_model,
            hidden_dim=int(mlp_ratio * d_model),
            dropout=0.0,
            act_layer=act_layer,
        )
        self.ln_2 = LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,  # (bs, slen, dim)
        mask: torch.Tensor = None,
    ): # --> (bs, slen, dim)
        """
        x.shape=(bsz*num_concurrent_media, num_chunks*ntks, dim)=(2, 4*1608, 1280)
        mask.shape=(2, 1, 4*1608, 4*1608)
        """
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()  # [-1, 1]
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        x = x + _gate_attn * self.attn(self.ln_1(x), mask=mask)
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x


class ImageTransformer(nn.Module):
    def __init__(
        self,
        width: int,   # 1280
        layers: int,  # 32
        heads: int,   # 16
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        self.width = width      # 1280
        self.layers = layers    # 32
        self.resblocks = nn.ModuleList(
            [
                ImageTransformerBlock(
                    d_model=width,          # 1280
                    n_head=heads,           # 16
                    mlp_ratio=mlp_ratio,    # 4.0
                    act_layer=act_layer,    # nn.GELU
                    gated=gated,            # False
                )
                for _ in range(self.layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_intermediate=None, mask=None):  # x: (bs, slen, dim)  --> (bs, slen, dim)
        """
        Example:
            x.shape=(bsz*num_concurrent_media, num_chunks*ntks, dim)=(2, 4*1608, 1280)
            return_intermediate=[3, 7, 15, 23, 30]
            mask.shape=(2, 1, 4*1608, 4*1608)
        """
        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask)
        if return_intermediate is not None:
            return x, torch.stack(out, dim=-1)  # (2, 4*1608, 1280), (2, 4*1608, 5, 1280)
        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        max_num_tiles: int,             # 4                  (Hardcoded)
        ckpt_path: str = None,
        image_size: int = 224,          # 560
        patch_size: int = 14,           # 14
        width: int = 1280,              # 1280      (Default)
        layers: int = 32,               # 32        (Default)
        heads: int = 16,                # 16        (Default)
        mlp_ratio: float = 4.0,         # 4.0       (Default)
        act_layer: Callable = nn.GELU,  # nn.GELU   (Default)
        in_channels: int = 3,           # 3         (Default)
        load_ckpt: bool = False,
        n_global_layers: int = 2,       # 8
        global_model: bool = False,     # True
        return_intermediate=None,       # [3, 7, 15, 23, 30] (Hardcoded)
    ):
        super().__init__()
        self.global_model = global_model                # True
        self.return_intermediate = return_intermediate  # [3, 7, 15, 23, 30]
        self.max_num_tiles = max_num_tiles              # 4
        self.image_size = to_2tuple(image_size)         # (560, 560)
        self.patch_size = to_2tuple(patch_size)         # (14, 14)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )                                               # (40, 40)
        self.conv1 = ColumnParallelConv2dPatch(
            in_channels=in_channels,  # 3
            out_channels=width,       # 1280
            kernel_size=patch_size,   # 14
            stride=patch_size,        # 14
            bias=False,
        )
        scale = width**-0.5  # 1280**-0.5=0.02795084971
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # (1280)
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )  # (1601, 1280)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.transformer = ImageTransformer(
            width, layers, heads, mlp_ratio, act_layer=act_layer  # 1280, 32, 16, 4.0, nn.GELU
        )
        # pre and post tile position embedding
        self.global_transformer = ImageTransformer(
            width, n_global_layers, heads, mlp_ratio, act_layer=act_layer, gated=True # 1280, 8, 16, 4.0, nn.GELU, True
        )
        # pre and post tile position embedding
        self.pre_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles,  # 4 # If a bigger number is used, it will do _dynamic_resize
            width=width,              # 1280
            gated=True,
        )
        self.post_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles, # 4 # If a bigger number is used, it will do _dynamic_resize
            width=width,             # 1280
            gated=True,
        )
        self.gated_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,                              # 4
                max_num_tiles,                              # 4
                self.grid_size[0] * self.grid_size[1] + 1,  # 1601
                width,                                      # 1280
            )
        )  # (4, 4, 1601, 1280)
        self.gated_positional_embedding_gate = nn.Parameter(torch.zeros(1))  # (1)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool = True,
        missing_keys: List[str] = None,
        unexpected_keys: List[str] = None,
        error_msgs: List[str] = None,
        return_state_dict: bool = False,
    ) -> None:
        orig_pos_embed = state_dict.get(prefix + "positional_embedding")
        if orig_pos_embed is not None:
            new_pos_embed = resize_local_position_embedding(
                orig_pos_embed, self.grid_size
            )
            state_dict[prefix + "positional_embedding"] = new_pos_embed
        if hasattr(self, "gated_positional_embedding"):
            if prefix + "gated_positional_embedding" not in state_dict:
                # resize positional_embedding to fit the new grid size
                global_pos_embed = initialize_global_position_embedding_from_local(
                    new_pos_embed,
                    self.grid_size,
                    self.max_num_tiles,
                    self.max_num_tiles,
                )
                state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
                state_dict[prefix + "gated_positional_embedding_gate"] = torch.zeros(
                    1, dtype=global_pos_embed.dtype
                )
                logger.info(
                    f"Initialized global positional embedding with size {global_pos_embed.size()}"
                )
            else:
                global_pos_embed = resize_global_position_embedding(
                    state_dict[prefix + "gated_positional_embedding"],
                    self.grid_size,
                    self.max_num_tiles,
                    self.max_num_tiles,
                )
                logger.info(
                    f"Resized global positional embedding from {state_dict[prefix + 'gated_positional_embedding'].size()} to {global_pos_embed.size()}"
                )
                state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
        if return_state_dict:
            return state_dict

    def apply_positional_embedding(self, x, ar):
        """
        x.shape = (bsz*num_concurrent_media, num_chunks, ntok, dim)  # (2, 4, 1601, 1280)
        ar.shape = (bsz*num_concurrent_media, 2)  # E.g., ar= [[1, 2], [2, 2]], shape=(2, 2)
        """
        out = []
        # apply regular position embedding
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)  # (8, 1601, 1280)
        x = x + self.positional_embedding * (
            1 - self.gated_positional_embedding_gate.tanh()
        )  # x: (8, 1601, 1280), y: (1601, 1280), x+y: (8, 1601, 1280)  # Grid level positional embedding
        x = x.view(bsz, num_chunks, num_tokens, dim)  # (2, 4, 1601, 1280)
        for idx, arx in enumerate(ar):
            # When idx=0, arx=[1, 2], _pos_embed.shape=(1, 2, 1601, 1280) 
            #   After reshape: (2, 1601, 1280)
            #   x[0, :2] += (_pos_embed * self.gated_positional_embedding_gate.tanh())
            # When idx=1, arx=[2, 2], _pos_embed.shape=(2, 2, 1601, 1280) --> (4, 1601, 1280)
            #  After reshape: (4, 1601, 1280)
            #   x[1, :4] += (_pos_embed * self.gated_positional_embedding_gate.tanh())
            _pos_embed = self.gated_positional_embedding[: arx[0], : arx[1]]
            _pos_embed = _pos_embed.reshape(arx[0] * arx[1], *_pos_embed.shape[2:])
            x[idx, : arx[0] * arx[1]] += (
                _pos_embed * self.gated_positional_embedding_gate.tanh()
            )
        return x  # (2, 4, 1601, 1280)

    def apply_class_embedding(self, x):
        # x: (bsz*num_concurrent_media*num_chunks, ntok, dim). E.g, (8, 1600, 1280)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),  # (1280) + (8, 1, 1280) --> (8, 1, 1280) # Copied the self.class_embedding to the last dimension of (bsz*num_concurrent_media*num_chunks, 1, 1280)
                x,  # (8, 1600, 1280)
            ],
            dim=1,
        )  # x: (8, 1601, 1280) # Appended the self.class_embedding as the last grid level vision_token of each chunk
        return x

    def forward(self, images: torch.Tensor, ar: torch.Tensor) -> torch.Tensor:
        """
        images.shape=(bsz, num_concurrent_media, num_chunks, nch, w, h). E.g., (1, 2, 4, 3, 560, 560).
            Remark: Here, num_chunks==max_num_chunks, and num_concurrent_media=num_images
        ar.shape= (bsz, num_images, 2_h_w) = (1, 2, 2). E.g., ar=tensor([[[1, 2], [2, 2]]])
            Remark: Here, ar means aspect_ratios

        Returns:
            x: (bsz, num_concurrent_media, num_chunks, ntok, -1) = (1, 2, 4, 1601, 5*1280)
            返回的x包含了第4, 8, 16, 24, 31层，以及最后一层，第40层（=32（transformer）+8（globaltransformer））的输出
        """
        if images.ndim == 5:
            num_concurrent_media = 1  # num_concurrent_media means the number of images before chunking in the input
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)  # (8, 3, 560, 560)
        ar = ar.reshape(bsz * num_concurrent_media, 2)                               # (2, 2)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)  # Remark. duplication: x = images
        x = self.conv1(x)  # (bsz*num_concurrent_media*num_chunks, num_vision_tokens, out_channels) = (8, 1600, 1280)
        _, ntok, dim = x.shape  # ntok: num_vision_tokens (=1600), dim: out_channels (=1280)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)  # (2, 4, 1600, 1280)

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar)  # (2, 4, 1600, 1280) # Embedded position information per chunk instead of per tile
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)  # (8, 1600, 1280)

        # apply "cls" token
        x = self.apply_class_embedding(x)  # (8, 1601, 1280)
        ntok += 1  # 1601

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)  # (2, 4, 1601, 1280)
        x = self.apply_positional_embedding(x, ar)  # (2, 4, 1601, 1280)  # Embedded position information per chunk per tile

        x = self.ln_pre(x)  # LayerNorm # (2, 4, 1601, 1280)
        npad, attn_mask = 0, None
        # Make sure the num_vision_tokens dimension is dividable by 8
        x, npad = expand_num_tokens_to_mult8(x)                                     # x.shape=(2, 4, 1608, 1280), npad=7 
        attn_mask = build_encoder_attention_mask(x, ar, ntok, num_chunks, 1)        # (2, 1, 4*1608, 4*1608)
        x = x.view(bsz * num_concurrent_media, -1, dim)                             # (2, 4*1608, 1280)
        x, int_x = self.transformer(    
            x, return_intermediate=self.return_intermediate, mask=attn_mask 
        )                                                                           # x: (2, 4*1608, 1280), int_x: (2, 4*1608, 5, 1280)

        x = self.ln_post(x)                                                         # LayerNorm # (2, 4*1608, 1280)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)     # (2, 4, 1608, 1280)
        x = self.post_tile_pos_embed(x, ar)                                         # (2, 4, 1608, 1280) # Embedded position information per chunk instead of per tile
        x = x.reshape(bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)  # (2, 4*1608, 1280)
        x = self.global_transformer(x, mask=attn_mask)                              # (2, 4*1608, 1280)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)     # (2, 4, 1608, 1280)
        x = contract_num_tokens_from_mult8(x, npad)                                 # (2, 4, 1601, 1280)

        # adding back intermediate layer outputs
        x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)                  # (1, 2, 4, 1601, 1280)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)   # (2, 4*1608, 5, 1280) --> (2, 4, 1608, 5*1280)
        int_x = contract_num_tokens_from_mult8(int_x, npad)                              # (2, 4, 1601, 5*1280)
        int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)           # (1, 2, 4, 1601, 5*1280)
        x = torch.cat([x, int_x], dim=-1)                                                # (1, 2, 4, 1601, 1280+5*1280)
        return x  # (1, 2, 4, 1601, 1280+5*1280)


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.
        Args:
            args (ModelArgs): Model configuration parameters.
        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.
        """
        super().__init__()
        model_parallel_size = fs_init.get_model_parallel_world_size()
        replication_factor = 1
        if model_parallel_size > 8:
            replication_factor = model_parallel_size // MP_SCALE

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_kv_heads *= replication_factor

        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_seq_len = args.max_seq_len

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # (n_local_heads*head_dim, args.dim)
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # (n_local_kv_heads*head_dim, args.dim)
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # (n_local_kv_heads*head_dim, args.dim)
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )  # (args.dim, n_local_heads*head_dim)
        self.n_heads = args.n_heads

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        cache_shape = (
            max_batch_size,
            self.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )
        device = next(self.parameters()).device
        self.register_buffer(
            "key_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        # print("Attention.forward (inputs)")
        # print(f"    x.shape={x.shape}")
        # print(f"    mask.shape={mask.shape}")
        # print(f"    freqs_cis.shape={freqs_cis.shape}")
        # print(f"    position_ids.shape={position_ids.shape}")
        # Attention.forward (inputs)
        #     x.shape               = torch.Size([1, 13, 4096])
        #     mask.shape            = torch.Size([1, 1, 13, 512])
        #     freqs_cis.shape       = torch.Size([13, 64])
        #     position_ids.shape    = torch.Size([13])

        xq, xk, xv = [
            F.linear(x, w) for w in [self.wq.weight, self.wk.weight, self.wv.weight]
        ]  
        # x: (bs, slen, dim), wq: (n_local_heads   *head_dimm, dim) -> xq: (bs, slen, n_local_heads   *head_dim)
        # x: (bs, slen, dim), wk: (n_local_kv_heads*head_dimm, dim) -> xk: (bs, slen, n_local_kv_heads*head_dim)
        # x: (bs, slen, dim), wv: (n_local_kv_heads*head_dimm, dim) -> xv: (bs, slen, n_local_kv_heads*head_dim)
        # print("After linear projections")
        # print(f"    xq.shape={xq.shape}")
        # print(f"    xk.shape={xk.shape}")
        # print(f"    xv.shape={xv.shape}")
        # After linear projections
        #     xq.shape=torch.Size([1, 13, 4096])
        #     xk.shape=torch.Size([1, 13, 1024])
        #     xv.shape=torch.Size([1, 13, 1024])

        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)            # xq: (bs, slen, n_local_heads,    head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)  # xk: (bs, slen, n_local_kv_heads, head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)  # xv: (bs, slen, n_local_kv_heads, head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        # print("After apply_rotary_emb")
        # print(f"    xq.shape={xq.shape}")
        # print(f"    xk.shape={xk.shape}")

        self.key_cache[:bs, position_ids, ...] = xk
        self.value_cache[:bs, position_ids, ...] = xv

        # TODO: we can avoid slicing on first dimension by always padding to max_batch_size()
        xk = self.key_cache[:bs, ...]
        xv = self.value_cache[:bs, ...]

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]
        # xq: (bs, n_local_heads,    slen, head_dim)
        # xk: (bs, n_local_kv_heads, slen, head_dim)
        # xv: (bs, n_local_kv_heads, slen, head_dim)

        xk = xk.repeat_interleave(self.n_rep, dim=1)  # xk: (bs, n_local_heads, slen, head_dim)
        xv = xv.repeat_interleave(self.n_rep, dim=1)  # xv: (bs, n_local_heads, slen, head_dim)

        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )  # attn_output: (bs, n_local_heads, slen, head_dim)
        # print("After attention")
        # print(f"    attn_output.shape={attn_output.shape}")

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)  # attn_output: (bs, slen, n_local_heads*head_dim)
        # print("After transpose and reshape")
        # print(f"    attn_output.shape={attn_output.shape}")
        out = F.linear(attn_output, self.wo.weight)  # attn_output: (bs, slen, n_local_heads*head_dim), self.wo.weight^T: (n_local_heads*head_dim, dim) -> out: (bs, slen, dim)
        # print("After output projection")
        # print(f"    out.shape={out.shape}")
        out = reduce_from_tensor_model_parallel_region(out)  # 'sum', out: (bs, slen, dim)
        # print(f"    out2.shape={out.shape}")
        
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,          # 4096
        hidden_dim: int,   # 4*4096 = 16384
        multiple_of: int,  # 1024
        ffn_dim_multiplier: Optional[float],  # 1.3
    ):
        """
        Initialize the FeedForward module.
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)  # 10922
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)  # 14198
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 14336

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        x1, x3 = [F.linear(x, w) for w in [self.w1.weight, self.w3.weight]]
        x1 = F.silu(x1)
        # print("FeedForward.forward")
        # print(f"    x1.shape={x1.shape} (after w1 and silu)")
        # print(f"    x3.shape={x3.shape} (after w3)")
        x_in = x1 * x3
        # print(f"    x_in.shape={x_in.shape}")
        out = F.linear(x_in, self.w2.weight)
        # print(f"    out.shape={out.shape}")
        out = reduce_from_tensor_model_parallel_region(out)
        # print(f"    out2.shape={out.shape}")

        # FeedForward.forward
        #     x1.shape      = torch.Size([1, 13, 14336]) (after w1 and silu)
        #     x3.shape      = torch.Size([1, 13, 14336]) (after w3)
        #     x_in.shape    = torch.Size([1, 13, 14336])
        #     out.shape     = torch.Size([1, 13, 4096])
        #     out2.shape    = torch.Size([1, 13, 4096])
        return out


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.
        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.
        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,                                # 4096
            hidden_dim=4 * args.dim,                     # 16384
            multiple_of=args.multiple_of,                # 1024
            ffn_dim_multiplier=args.ffn_dim_multiplier,  # 1.3
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.attention.setup_cache(max_batch_size, dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the TransformerBlock.
        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        # print("TransformerBlock.forward (inputs)")
        # print(f"    x.shape={x.shape}")
        # print(f"    freqs_cis.shape={freqs_cis.shape}")
        # print(f"    mask.shape={mask.shape}")   
        # print(f"    position_ids.shape={position_ids.shape}")

        # TransformerBlock.forward (inputs)
        # x.shape            = torch.Size([1, 13, 4096])
        # freqs_cis.shape    = torch.Size([13, 64])
        # mask.shape         = torch.Size([1, 1, 13, 512])
        # position_ids.shape = torch.Size([13])

        h = self.attention.forward(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            position_ids=position_ids,
        )
        # print("After attention")
        # print(f"    h.shape={h.shape}")
        # After attention
        #     h.shape=torch.Size([1, 13, 4096])
        h = h + x
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class TilePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_tiles: int,  # 4 (max_num_tiles)
        width: int,      # 1280
        gated: bool = False,
    ):
        super().__init__()
        self.num_tiles = num_tiles  # 4
        self.width = width          # 1280 
        self.embedding = nn.Parameter(
            torch.randn(num_tiles, num_tiles, 1, width) / math.sqrt(width)
        )                           # (4, 4, 1, 1280)
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # load the weights from the checkpoint
        embed = state_dict.get(prefix + "embedding")
        if embed is not None:
            # reshape the weights to the correct shape
            nt_old, nt_old, _, w = embed.shape
            logging.info(
                f"Resizing tile embedding from {nt_old}x{nt_old} to {self.num_tiles}x{self.num_tiles}"
            )
            embed_new = TilePositionEmbedding._dynamic_resize(embed, self.num_tiles)
            # assign the weights to the module
            state_dict[prefix + "embedding"] = embed_new

    @staticmethod
    def _dynamic_resize(embed: torch.Tensor, num_tiles: int):
        nt_old, nt_old, _, w = embed.shape
        embed = embed.permute(2, 3, 0, 1)

        embed_new = F.interpolate(
            embed,
            size=(num_tiles, num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embed_new = embed_new.permute(2, 3, 0, 1)
        return embed_new

    def forward(self, x: torch.Tensor, ar: torch.Tensor, num_tiles: int = None):
        """
        x: shape=(bsz * num_concurrent_media, num_chunks, ntok, dim). E.g, (2, 4, 1600, 1280)
        ar: shape=(bsz * num_concurrent_media, 2). E.g, ar=[[1, 2], [2, 2]], with shape=(2, 2)
        """
        embed = self.embedding  # (4, 4, 1, 1280)
        if num_tiles is None:
            num_tiles = self.num_tiles  # 4
        elif num_tiles > self.num_tiles:
            embed = TilePositionEmbedding._dynamic_resize(self.embedding, num_tiles)
        out_pos_embed = torch.zeros(
            x.shape[0], num_tiles, 1, self.width, device=x.device, dtype=x.dtype
        )  # (2, 4, 1, 1280)
        for idx, arx in enumerate(ar):
            h, w = arx  # h=1, w=2 or h=2, w=2
            out_pos_embed[idx, : w * h] = embed[:h, :w].reshape(w * h, 1, self.width)
            # out_pos_embed[0, : 2] = embed[:1, :2].reshape(2, 1, 1280) when idx=0
            # out_pos_embed[1, : 4] = embed[:2, :2].reshape(4, 1, 1280) when idx=1
        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()
        x = x + out_pos_embed  # x: (2, 4, 1600, 1280), out_pos_embed: (2, 4, 1, 1280) --> x + out_pos_embed: (2, 4, 1600, 1280)
        return x  # (2, 4, 1600, 1280)


def _noinit(x):
    return x


class CrossAttention(torch.nn.Module):
    """Cross attention layer with model-parallel attention layers."""

    def __init__(
        self,
        dim: int,           # 4096
        head_dim: int,      # 128
        n_heads: int,       # 32
        n_kv_heads: int,    # 16
        norm_eps: float,    # 1e-5
    ):
        super().__init__()
        self.model_parallel_size = fs_init.get_model_parallel_world_size()
        replication_factor = 1
        if self.model_parallel_size > 8:
            replication_factor = self.model_parallel_size // MP_SCALE
        n_kv_heads *= replication_factor

        assert n_heads % n_kv_heads == 0

        self.wq = ColumnParallelLinear(
            dim,
            n_heads * head_dim,
            bias=False,
            gather_output=False,
            init_method=_noinit,
        )  # wq: (n_local_heads*head_dim, dim)= (32*128/model_parallel_size, 4096)

        self.wk = ColumnParallelLinear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
            gather_output=False,
            init_method=_noinit,
        )  # wk: (n_local_kv_heads*head_dim, dim) = (16*128/model_parallel_size, 4096)
        self.wv = ColumnParallelLinear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
            gather_output=False,
            init_method=_noinit,
        )  # wv: (n_local_kv_heads*head_dim, dim) = (16*128/model_parallel_size, 4096)
        self.wo = RowParallelLinear(
            n_heads * head_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=_noinit,
        )  # wo: (dim, n_local_heads*head_dim) = (4096, 32*128/model_parallel_size)

        self.n_heads = n_heads          # 32
        self.head_dim = head_dim        # 128
        self.n_kv_heads = n_kv_heads    # 16

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )

        # cross-attention heads are model parallel similar to
        # self-attention, and we also use the identical KV head
        # combination to ensure parity with the corresponding
        # trunk LLM (i.e., group query attention) -- @dubeya
        # local heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_heads % self.model_parallel_size == 0
        assert self.n_kv_heads % self.model_parallel_size == 0
        self.n_local_heads = self.n_heads // self.model_parallel_size       
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

    def _compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        """
        Example: xattn_tokens.shape=(bsz, num_concurrent_media*max_num_chunks*ntok, image_token_dim)=(bsz, seqlen_y, dim)=(1, 2*4*1601, 4096)

        Returns:
            xk.shape=(bsz, n_local_heads, seqlen_y, head_dim)=(1, 32/model_parallel_size, 2*4*1601, 128)
            xv.shape=(bsz, n_local_heads, seqlen_y, head_dim)=(1, 32/model_parallel_size, 2*4*1601, 128)
        """
        bsz = xattn_tokens.shape[0]
        xk = self.wk(xattn_tokens)  # xattn_tokens: (bsz, seqlen_y, dim), wk.weight^T: (dim, n_local_kv_heads*head_dim) -> xk: (bsz, seqlen_y, n_local_kv_heads*head_dim)
        xv = self.wv(xattn_tokens)  # xattn_tokens: (bsz, seqlen_y, dim), wv.weight^T: (dim, n_local_kv_heads*head_dim) -> xv: (bsz, seqlen_y, n_local_kv_heads*head_dim)

        _, seqlen_y, _ = xk.shape

        xk = xk.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)  # xk: (bsz, seqlen_y, n_local_kv_heads, head_dim)
        xv = xv.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)  # xv: (bsz, seqlen_y, n_local_kv_heads, head_dim)

        xk, xv = [tensor.transpose(1, 2) for tensor in (xk, xv)]
        # xk: (bsz, n_local_kv_heads, seqlen_y, head_dim)
        # xv: (bsz, n_local_kv_heads, seqlen_y, head_dim)

        # repeat k/v heads if n_kv_heads < n_heads
        xk = xk.repeat_interleave(self.n_rep, dim=1)  # xk: (bsz, n_local_heads, seqlen_y, head_dim)
        xv = xv.repeat_interleave(self.n_rep, dim=1)  # xv: (bsz, n_local_heads, seqlen_y, head_dim)

        xk = self.k_norm(xk)

        return torch.stack([xk, xv])

    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self._compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        Example:
            Inputs:
        """
        # print("CrossAttention.forward")
        # print(f"    x.shape={x.shape}")
        # print(f"xattn_cache: {xattn_cache.shape}, x: {x.shape}")
        xq = F.linear(x, self.wq.weight)  # x: (bsz, seqlen, dim), wq.weight^T: (dim, n_local_heads*head_dim) -> xq: (bsz, seqlen, n_local_heads*head_dim)
        bsz, seqlen, _ = x.shape
        # print(f"    xq.shape={xq.shape} (after linear projection)")

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)  # xq: (bsz, seqlen, n_local_heads, head_dim)
        # print(f"    xq.shape={xq.shape} (after view)")
        xq = self.q_norm(xq)
        # print(f"    xq.shape={xq.shape} (after q_norm)")
        xq = xq.transpose(1, 2)  # xq: (bsz, n_local_heads, seqlen, head_dim)
        # print(f"    xq.shape={xq.shape} (after transpose)")

        xk, xv = xattn_cache  # xk: (bsz, n_local_heads, seqlen_y, head_dim), xv: (bsz, n_local_heads, seqlen_y, head_dim)
        # print(f"xk: {xk.shape}, xv: {xv.shape}, xq: {xq.shape}")
        # print(f"    xk.shape= {xk.shape} (from xattn_cache)")
        # print(f"    xv.shape= {xv.shape} (from xattn_cache)")

        # Explanation of the following "scaled_dot_product_attention" computation:
        # attn_score = xq @ xk^T: (bsz, n_local_heads, seqlen, head_dim) @ (bsz, n_local_heads, head_dim, seqlen_y) -> (bsz, n_local_heads, seqlen, seqlen_y)
        # output = attn_score @ xv: (bsz, n_local_heads, seqlen, seqlen_y) @ (bsz, n_local_heads, seqlen_y, head_dim) -> (bsz, n_local_heads, seqlen, head_dim)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=xattn_mask, dropout_p=0.0
        )  # output: (bsz, n_local_heads, seqlen, head_dim)
        output = output * full_text_row_masked_out_mask
        # print(f"    output.shape={output.shape} (after scaled_dot_product_attention)")
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)  # output: (bsz, seqlen, n_local_heads*head_dim)
        # print(f"    output.shape={output.shape} (after transpose and reshape)")
        out = F.linear(output, self.wo.weight)  # output: (bsz, seqlen, n_local_heads*head_dim), self.wo.weight^T: (n_local_heads*head_dim, dim) -> out: (bsz, seqlen, dim)
        # print(f"    out.shape={out.shape} (after output projection)")
        out = reduce_from_tensor_model_parallel_region(out)  # out: (bsz, seqlen, dim)
        # print(f"    out2.shape={out.shape} (after reduce_from_tensor_model_parallel_region)")

        # CrossAttention.forward
        #     x.shape       = torch.Size([1, 13, 4096])
        #     xq.shape      = torch.Size([1, 13, 4096])         (after linear projection)
        #     xq.shape      = torch.Size([1, 13, 32, 128])      (after view)
        #     xq.shape      = torch.Size([1, 13, 32, 128])      (after q_norm)
        #     xq.shape      = torch.Size([1, 32, 13, 128])      (after transpose)
        #     xk.shape      = torch.Size([1, 32, 6404, 128])    (from xattn_cache)
        #     xv.shape      = torch.Size([1, 32, 6404, 128])    (from xattn_cache)
        #     output.shape  = torch.Size([1, 32, 13, 128])      (after scaled_dot_product_attention)
        #     output.shape  = torch.Size([1, 13, 4096])         (after transpose and reshape)
        #     out.shape     = torch.Size([1, 13, 4096])         (after output projection)
        #     out2.shape    = torch.Size([1, 13, 4096])         (after reduce_from_tensor_model_parallel_region)
        return out


class CrossAttentionTransformerBlock(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(
        self,
        args: ModelArgs,
        layer_id: int,
        no_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads         # 32
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads   # 8
        self.dim = args.dim                 # 4096
        self.head_dim = args.dim // args.n_heads   # 128
        self.attention = CrossAttention(
            dim=args.dim,                   # 4096
            head_dim=self.head_dim,         # 128
            n_heads=self.n_heads,           # 32
            n_kv_heads=self.n_kv_heads,     # 8
            norm_eps=args.norm_eps,         # 1e-5
        )

        self.attention_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
        )
        self.gate_attn = torch.nn.Parameter(torch.zeros(1))

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            multiple_of=args.multiple_of,
        )
        self.ffn_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
        )
        self.gate_ffwd = torch.nn.Parameter(torch.zeros(1))

        self.no_ffn = no_ffn

    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self.attention.compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        xattn_cache: torch.Tensor,
    ) -> torch.Tensor:
        _attn_out = self.attention(
            x=self.attention_norm(x),
            xattn_mask=xattn_mask,
            xattn_cache=xattn_cache,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )
        # print("CrossAttentionTransformerBlock.forward")
        # print(f"    _attn_out.shape={_attn_out.shape} (after cross attention)")
        h = x + self.gate_attn.tanh() * _attn_out
        # print(f"    h.shape={h.shape}")
        _ffn = self.feed_forward(self.ffn_norm(h))
        _ffn = full_text_row_masked_out_mask[:, 0] * _ffn  # type: ignore
        # print(f"    full_text_row_masked_out_mask[:, 0].shape: {full_text_row_masked_out_mask[:, 0].shape}")
        # print(f"    _ffn.shape={_ffn.shape} (after full_text_row_masked_out_mask)")
        h = h + self.gate_ffwd.tanh() * _ffn * float(not self.no_ffn)
        # print(F"    h.shape={h.shape}")

        # CrossAttentionTransformerBlock.forward
        #     _attn_out.shape   =   torch.Size([1, 13, 4096]) (after cross attention)
        #     h.shape           =   torch.Size([1, 13, 4096])
        #     full_text_row_masked_out_mask[:, 0].shape = torch.Size([1, 13, 1])
        #     _ffn.shape        =   torch.Size([1, 13, 4096]) (after full_text_row_masked_out_mask)
        #     h.shape           =   torch.Size([1, 13, 4096])
        return h


class DummyCrossAttentionTransformerBlock:
    """Dummy cross-attention transformer block with tanh-gated attention and feedforward."""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


class DummySelfAttentionTransformerBlock:
    """Dummy self-attention transformer block"""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


class CrossAttentionTransformerVision(torch.nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        return_intermediate = "3,7,15,23,30"
        self.vision_input_dim = 1280
        self.image_res = args.vision_chunk_size             # 560
        self.max_num_chunks = args.vision_max_num_chunks    # 4
        if return_intermediate is not None:
            return_intermediate = [int(l) for l in return_intermediate.split(",")]  # [3, 7, 15, 23, 30]
            self.vision_input_dim = (
                len(return_intermediate) + 1
            ) * self.vision_input_dim                 # (5+1)*1280
        self.patch_size = 14
        self.vision_encoder = VisionEncoder(
            max_num_tiles=4,  # Remark: Hardcoded to 4. If a bigger number is needed, it will do _dynamic_resize.
            image_size=args.vision_chunk_size,        # 560
            patch_size=self.patch_size,               # 14
            n_global_layers=8,                        # Hardcoded: 8
            global_model=True,                        # Hardcoded: True
            return_intermediate=return_intermediate,  # Hardcoded: [3, 7, 15, 23, 30]
        )
        # vision token projection
        self.vision_projection = ColumnParallelLinear(
            self.vision_input_dim,  # 6*1280
            args.dim,               # 4096
            bias=True,
            init_method=lambda x: x,
        )  # (local_dim, vision_input_dim) = (4096/model_parallel_size, 1280)

    def forward(
        self, images: torch.Tensor, aspect_ratios: torch.Tensor
    ) -> torch.Tensor:
        """
        Example 
            Inputs:
                image (stacked_images): shape: (bsz, num_images, max_num_chunks, 3, 560, 560) = (1, 2, 4, 3, 560, 560)
                aspect_ratios: tensor([[[1, 2], [2, 2]]]), shape: (bsz, num_images, 2_h_w) = (1, 2, 2)
            Outputs:
                vision_tokens: shape: (bsz, num_images, max_num_chunks, 1601, 4096) = (1, 2, 4, 1601, 4096)
        """
        # vision_tokens: (B, T, D)
        # aspect_ratios: (B, T)
        # h: (B, T, D)
        vision_tokens = self.vision_encoder(
            images.to(dtype=torch.bfloat16), aspect_ratios  # images.shape=(1, 2, 4, 3, 560, 560), aspect_ratios.shape=(1, 2, 2)
        ) # vision_tokens.shape=(1, 2, 4, 1601, 1280+5*1280)

        # vision_tokens.shape = (1, 2, 4, 1601, 6*1280)
        # vision_projection.weight^T.shape = (6*1280, 4096/model_parallel_size)
        vision_tokens = F.linear(
            vision_tokens, self.vision_projection.weight, self.vision_projection.bias
        ) # vision_tokens.shape=(1, 2, 4, 1601, 4096/model_parallel_size)
        vision_tokens = gather_from_tensor_model_parallel_region(vision_tokens)  # vision_tokens.shape=(1, 2, 4, 1601, 4096)
        return vision_tokens  # (1, 2, 4, 1601, 4096)


class CrossAttentionTransformerText(torch.nn.Module):
    INFERENCE_IMAGE_TOKEN_ID = 128010

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_parallel_size = fs_init.get_model_parallel_world_size()
        print(f"model_parallel_size: {self.model_parallel_size}")
        assert args.vocab_size > 0
        self.vocab_size = args.vocab_size  # 128256
        self.n_layers = args.n_layers      # 32
        self.dim = args.dim                # 4096
        self.head_dim = args.dim // args.n_heads  # 4096/32=128
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # 8
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        assert self.vocab_size % self.model_parallel_size == 0
        # A set of embeddings that are not updated during training (they are "frozen").
        # 128000*4096*2 / (1024*1024*1024) = 1.5GB
        self.tok_embeddings = VocabParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )  # (local_vocab_size,embedding_dim)=(args.vocab_size/model_parallel_size, args.dim)
        self.pos_embeddings = None
        # final norm layer (not necessary for post-norm)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # output layer
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )  # output: (local_vocab_size, args.dim)

        self.n_llama_layers = args.n_layers
        self.model_dim = args.dim

        # BLOCKS

        self.fusion_schedule = self._init_fusion_schedule(
            args.vision_num_cross_attention_layers
        )
        # A set of embeddings that are updated during training (they are "learnable").
        self.learnable_embedding = VocabParallelEmbedding(
            max(fs_init.get_model_parallel_world_size(), 8),
            args.dim,
            init_method=lambda x: x,
        )  # (max(fs_init.get_model_parallel_world_size(),8)/model_parallel_size, args.dim)
        self.num_frozen_embeddings = self.tok_embeddings.num_embeddings  # args.vocab_size = 128256
        self._thresh = self.num_frozen_embeddings - 1  # 128255

        # transformer blocks
        self.layers = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()
        for i in range(args.n_layers):
            layer_id = i
            block = TransformerBlock(args=args, layer_id=layer_id)
            self.layers.append(block)
            if layer_id in self.fusion_schedule:
                xa_layer_id = self.fusion_schedule.index(layer_id) + args.n_layers
                block = CrossAttentionTransformerBlock(
                    args,
                    layer_id=xa_layer_id,
                )
                self.cross_attention_layers.append(block)

        # add xattn and dummy layers to avoid conditionals in forward()
        self.text_and_xattn_layers = []

        for idx, layer in enumerate(self.layers):
            if idx in self.fusion_schedule:
                xattn_layer_idx = self.fusion_schedule.index(idx)
                xattn_layer = self.cross_attention_layers[xattn_layer_idx]
            else:
                xattn_layer_idx = 0
                xattn_layer = DummyCrossAttentionTransformerBlock()

            self.text_and_xattn_layers.append(
                (
                    layer,
                    xattn_layer,
                    xattn_layer_idx,
                )
            )
        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
            args.use_scaled_rope,
        )

        self.args = args
        self.cache_is_setup = False
        self.max_seq_len = args.max_seq_len

    def _init_fusion_schedule(
        self,
        num_layers: int,  # 8
    ) -> List[int]:
        llama_layers = list(range(self.n_llama_layers))  # [0,1,2,3,...,31]

        # uniformly spread the layers
        k = math.ceil(len(llama_layers) / num_layers)  # 32/8=4
        return llama_layers[::-1][::k][:num_layers][::-1]  # [3, 7, 11, 15, 19, 23, 27, 31]

    def get_partially_trainable_embedding(self, x):
        """
        Example:
            Inputs: x.shape=(bsz, seqlen)=(1, 21)
        """
        # Let's assume x.shape = (bsz, seqlen)
        xz = torch.zeros_like(x, device=x.device)  # xz: (bsz, seqlen)
        oz = torch.ones_like(x, device=x.device)   # oz: (bsz, seqlen)
        x_orig = torch.minimum(x, torch.tensor(self._thresh, device=x.device))  # Filter/replace the image token. x_orig: (bsz, seqlen). If x >= 128255, x_orig=128255, else x_orig=x
        x_new = (
            torch.maximum(x, torch.tensor(self._thresh + 1, device=x.device))
            - self.num_frozen_embeddings
        )  # x_new: (bsz, seqlen). If x < 128255, x_new=0, else x_new=x-128255

        mask_orig = torch.where(x >= self.num_frozen_embeddings, xz, oz).unsqueeze(-1)  # mask_orig: (bsz, seqlen, 1). If x >= 128255, mask_orig=0.0, else mask_orig=1.0
        mask_new = torch.where(x < self.num_frozen_embeddings, xz, oz).unsqueeze(-1)    # mask_new : (bsz, seqlen, 1). If x < 128255, mask_new=0.0, else mask_new=1.0

        x_orig = self.tok_embeddings(x_orig)  # Frozen embeddings. x_orig: (bsz, seqlen, dim)
        x_new = self.learnable_embedding(x_new).type_as(x_orig)  # Trainable embeddings. x_new: (bsz, seqlen, dim)
        return x_orig * mask_orig.type_as(x_orig) + x_new * mask_new.type_as(x_new)  # (bsz, seqlen, dim)

    def forward(
        self,
        position_ids: torch.LongTensor,
        h: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_caches: torch.Tensor,
        text_only_inference: bool = False,
    ):
        assert self.cache_is_setup, "Please set up cache before calling forward"
        # print("CrossAttentionTransformerText.forward (inputs)")
        # print(f"    position_ids: {position_ids.shape}") 
        # print(f"    h: {h.shape}")
        # print(f"    xattn_mask: {xattn_mask.shape}")
        # print(f"    full_text_row_masked_out_mask: {full_text_row_masked_out_mask.shape}")
        # print(f"    xattn_caches: {xattn_caches.shape}")
        # print(f"    text_only_inference: {text_only_inference}")

        # CrossAttentionTransformerText.forward (inputs)
        # position_ids                  : torch.Size([13])
        # h                             : torch.Size([1, 13, 4096])
        # xattn_mask                    : torch.Size([1, 1, 13, 6404])
        # full_text_row_masked_out_mask : torch.Size([1, 1, 13, 1])
        # xattn_caches                  : torch.Size([8, 2, 1, 32, 6404, 128]) # (num_xattn_layers, 2_k_v, bsz, n_local_heads, seqlen, head_dim)
        # text_only_inference           : False
        # mask                          : torch.Size([1, 1, 13, 512])
        # freqs_cis                     : torch.Size([13, 64])

        mask = self.mask_cache.index_select(2, position_ids) 
        freqs_cis = self.freqs_cis.index_select(0, position_ids)
        # print(f"    mask: {mask.shape}")
        # print(f"    self.freqs_cis: {self.freqs_cis.shape}")
        # print(f"    freqs_cis: {freqs_cis.shape}")

        # for i in range(13):
        #     print(mask[0, 0, i, :15])

        for idx, (
            layer,
            xattn_layer,
            xattn_layer_idx,
        ) in enumerate(self.text_and_xattn_layers):
            # if not text_only_inference:
            h = xattn_layer(
                x=h,
                xattn_mask=xattn_mask,
                xattn_cache=xattn_caches[xattn_layer_idx],
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            )
            h = layer(
                x=h,
                mask=mask,
                freqs_cis=freqs_cis,
                position_ids=position_ids,
            )

        h = self.norm(h)

        output = F.linear(h, self.output.weight)
        output = gather_from_tensor_model_parallel_region(output)
        return output.float()

    def setup_cache(self, max_batch_size: int, dtype=torch.bfloat16):
        # Set up the text kv caches
        device = next(self.parameters()).device
        ones = torch.ones(
            (self.max_seq_len, self.max_seq_len),
            dtype=torch.bool,
            device=device,
        )
        self.register_buffer(
            "mask_cache",
            torch.tril(
                ones,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            persistent=False,
        )
        for layer in self.layers:
            layer.setup_cache(max_batch_size, dtype=dtype)
        self.cache_is_setup = True

    def _get_xattn_mask(
        self,
        num_tokens,
        text_device,
        text_dtype,
        vision_tokens,
        cross_attention_masks,
    ) -> Tuple[Tensor, Tensor]:
        """
        Example:
            Inputs:
                num_tokens= 512           
                text_device="cuda"
                text_dtype=torch.bfloat16
                vision_tokens=a tensor with shape (bsz, num_images, max_num_chunks, num_image_tokens, image_token_dim),
                    where num_image_tokens=1600_num_tiles+1_CLS. E.g., (1, 2, 4, 1601, 4096)
                cross_attention_masks= padded_masks, such that 
                    padded_masks.shape (1, 512, 2, 4)  # (bsz, num_tokens, num_images, max_num_chunks)
                    padded_masks[0, 10:20, 0, :2] = 0.0
                    padded_masks[0, 20:21, 1, :4] = 0.0
                    padded_masks[otherwise] = -inf
            Returns:
                cross_attention_masks: (1, 1, 512, 2*4*1601)
                full_text_row_masked_out_mask: (1, 1, 512, 2*4*1601)
        
        """
        assert vision_tokens is not None, "Vision tokens must be provided"
        vision_seqlen = vision_tokens.shape[3]  # 1601
        assert (
            vision_tokens.shape[1] == cross_attention_masks.shape[2]
        ), f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_masks.shape}"
        assert (
            vision_tokens.shape[2] == cross_attention_masks.shape[3]
        ), f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_masks.shape}"
        assert (
            num_tokens == cross_attention_masks.shape[1]
        ), f"Mismatch in text sequence length and cross attention mask sequence length {num_tokens} {cross_attention_masks.shape}"
        _, _, _, num_image_tokens, image_token_dim = tuple(vision_tokens.shape)
        bsz, ntext, nimg, nchunks = cross_attention_masks.shape
        cross_attention_masks = (
            cross_attention_masks.repeat_interleave(vision_seqlen, dim=3)  # (bsz, ntext, nimg, nchunks*vision_seqlen)=(1, 512, 2, 4*1601)
            .view(bsz, ntext, -1)   # (bsz, ntext, nimg*nchunks*vision_seqlen)=(1, 512, 2*4*1601)
            .unsqueeze(1)           # (bsz, 1, ntext, nimg*nchunks*vision_seqlen)=(1, 1, 512, 2*4*1601)
        )
        full_text_row_masked_out_mask = _get_full_row_masked_out_mask(
            cross_attention_masks,  # (bsz, 1, ntext, nimg*nchunks*vision_seqlen)=(1, 1, 512, 2*4*1601)
            get_negative_inf_value(cross_attention_masks.dtype),
        )
        cross_attention_masks *= full_text_row_masked_out_mask

        return (
            cross_attention_masks.to(device=text_device, dtype=text_dtype),  # (1, 1, 512, 2*4*1601)
            full_text_row_masked_out_mask,                                   # (1, 1, 512, 2*4*1601)
        )


class CrossAttentionTransformer(torch.nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.params = args

        self.model_dim = args.dim  # 4096
        self.vision_model = CrossAttentionTransformerVision(args)
        self.text_model = CrossAttentionTransformerText(args)
        self.image_res = args.vision_chunk_size  # 560
        self.max_num_chunks = args.vision_max_num_chunks  # 4
        self.image_transform = partial(
            VariableSizeImageTransform(size=args.vision_chunk_size),
            max_num_chunks=args.vision_max_num_chunks,
        )

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.text_model.setup_cache(max_batch_size, dtype)

    def compute_vision_tokens_masks(
        self,
        batch_images: List[List[PIL_Image.Image]],  # [[<PIL.Image.Image object at 0x...>, <PIL.Image.Image object at 0x...>]]
        batch_masks: List[List[List[int]]],         # [[[10, 20], [20, 21]]]
        total_len: int,                             # 512
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Example:
            batch_images = [[<PIL.Image.Image object at 0x...>, <PIL.Image.Image object at 0x...>]]  # img_1.shape=(300, 800), image_2.shape=(1024, 800)
            batch_masks  = [[[10, 20], [20, 21]]]
        """
        skip_vision_encoder = False

        assert len(batch_images) == len(
            batch_masks
        ), "Images and masks must have the same length"

        max_num_images = max(len(x) for x in batch_images)  # 2
        bsz = len(batch_images)                             # 1

        if max_num_images == 0:
            num_chunks = [[self.max_num_chunks] for _ in batch_images]
            skip_vision_encoder = True
        else:
            images_and_aspect_ratios = [
                [self.image_transform(im) for im in row] for row in batch_images
            ]  # [[[Tensor with shape 2_chunksx3x560x560, (h_ratio_1, w_ratio_2)], [Tensor with shape 4_chunksx3x560x560, (h_ratio_2, w_ratio_2)]]]
            transformed_images = [
                [x[0] for x in row] for row in images_and_aspect_ratios
            ]  # [[Tensor with shape 2_chunksx3x560x560, Tensor with shape 4_chunksx3x560x560]]

            aspect_ratios = torch.ones(bsz, max_num_images, 2, dtype=torch.int64)  # (bsz, max_num_images, 2) = (1, 2, 2)
            for i, row in enumerate(images_and_aspect_ratios):
                if len(row) > 0:
                    aspect_ratios[i, : len(row)] = torch.stack(
                        [torch.tensor(x[1]) for x in row]
                    )
            # aspect_ratios: shape (1, 2, 2) tensor([[[1, 2], [2, 2]]]), indicating that the aspect ratio of the first image is (1, 2) and the second image is (2, 2)

            stacked_images, num_chunks = _stack_images(
                transformed_images,
                max_num_chunks=self.max_num_chunks,
                image_res=self.params.vision_chunk_size,
                max_num_images=max_num_images,
            )
            # stacked_images: (bsz, max_num_images, max_num_chunks, 3, 560, 560) = (1, 2, 4, 3, 560, 560)
            #   Content: 
            #     Batch 0, Image 0: The first 2 chunks contains the image data. The remaining 2 chunks are zeros (padding)
            #              Image 1: The first 4 chunks contains the image data. No padding is necessary.
            # num_chunks: [[2, 4]] 
            #   Content: Batch 0: Image 0 has 2 chunks. Image 1 has 4 chunks.

        if skip_vision_encoder:
            vision_tokens = torch.zeros(
                (
                    bsz,
                    max_num_images,
                    self.max_num_chunks,  # 4
                    int(
                        (self.vision_model.image_res / self.vision_model.patch_size)  # 560/14=40
                        ** 2
                        + 1
                    ),  # 1601
                    self.model_dim,
                ),
            )
        else:
            vision_tokens = self.vision_model(stacked_images, aspect_ratios)  
            # (1, 2, 4, 1601, 4096) = (bsz, num_concurrent_media, max_num_chunks, ntok, dim), dim=image_token_dim

        vision_tokens = vision_tokens.to("cuda")

        bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)
        xattn_caches = torch.stack(
            [
                layer.compute_xattn_kv_cache(
                    vision_tokens.view(bsz, -1, image_token_dim)  # (1, 2*4*1601, 4096)
                )  # [xk.shape, xv.shape]=[(1, 32/model_parallel_size, 2*4*1601, 128), (1, 32/model_parallel_size, 2*4*1601, 128)]
                for layer in self.text_model.cross_attention_layers
            ]
        )  # (8, 2, 32/model_parallel_size, 2*4*1601, 128)
        padded_masks = _pad_masks(
            batch_masks,            # [[[10, 20], [20, 21]]]
            num_chunks,             # [[2, 4]]
            total_len,              # 512
            self.max_num_chunks,    # 4
        )
        # padded_masks.shape = (1, 512, 2, 4), such that
        #    padded_masks[0, 10:20, 0, :2] = 0.0
        #    padded_masks[0, 20:21, 1, :4] = 0.0
        #    padded_masks[otherwise] = -inf

        cross_attention_masks, full_text_row_masked_out_mask = (
            self.text_model._get_xattn_mask(
                num_tokens=total_len,                                   # 512           
                text_device="cuda",                                     # "cuda" (Hardcoded)
                text_dtype=next(self.text_model.parameters()).dtype,    # torch.bfloat16
                vision_tokens=vision_tokens,                            # shape: (1, 2, 4, 1601, 4096)
                cross_attention_masks=padded_masks,                     # shape: (1, 512, 2, 4)
            )
        )  # (1, 1, 512, 2*4*1601), (1, 1, 512, 2*4*1601)
        # print("CrossAttentionTransformer.compute_vision_tokens_masks")
        # print(f"    xattn_cache.shape: {xattn_caches.shape}")
        # print(f"    cross_attention_masks.shape: {cross_attention_masks.shape}")
        # print(f"    full_text_row_masked_out_mask.shape: {full_text_row_masked_out_mask.shape}")

        # xattn_caches.shape: (8, 2, 1, 32/model_parallel_size, 2*4*1601, 128)
        # cross_attention_masks.shape: (1, 1, 512, 2*4*1601)
        # full_text_row_masked_out_mask.shape: (1, 1, 512, 2*4*1601)

        # Example:
        # xattn_cache.shape: torch.Size([8, 2, 1, 32, 6404, 128])
        # cross_attention_masks.shape: torch.Size([1, 1, 512, 6404])
        # full_text_row_masked_out_mask.shape: torch.Size([1, 1, 512, 1])
        return (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask)

    def forward(
        self,
        position_ids: torch.Tensor,
        tokens: torch.Tensor,
        cross_attention_masks: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_caches: torch.Tensor,
        text_only_inference: bool = False,
    ) -> torch.Tensor:
        """
        Example:
            Inputs:
                position_ids: tensor([0, 1, 2, 3, ..., 20])
                tokens.shape: (bsz, total_len) = (1, 512). In our example, the first 21 elements contains meaningful tokens. The rest are padding.
                cross_attention_masks.shape: (1, 1, 512, 2*4*1601)
                full_text_row_masked_out_mask.shape: (1, 1, 512, 2*4*1601)
                xattn_caches.shape: (8, 2, 32/model_parallel_size, 2*4*1601, 128)
        """
        h = self.text_model.get_partially_trainable_embedding(tokens[:, position_ids])
        # print("CrossAttentionTransformer.forward (before text_model.forward)")
        # print(f"    position_ids: {position_ids.shape}")
        # print(f"    h.shape: {h.shape}")
        # print(f"    xattn_mask.shape: {cross_attention_masks.shape} <-- cross_attention_masks")
        # print(f"    full_text_row_masked_out_mask.shape: {full_text_row_masked_out_mask.shape}")
        # print(f"    xattn_caches.shape: {xattn_caches.shape}")
        # print(f"    text_only_inference: {text_only_inference}")
        # Example: h.shape: (1, 21, 4096)
        logits = self.text_model.forward(
            position_ids=position_ids,
            h=h,
            xattn_mask=cross_attention_masks[:, :, position_ids],
            full_text_row_masked_out_mask=full_text_row_masked_out_mask[
                :, :, position_ids
            ],
            xattn_caches=xattn_caches,
            text_only_inference=text_only_inference,
        )
        return logits


def _stack_images(
    images: List[List[PIL_Image.Image]],
    max_num_chunks: int,
    image_res: int,
    max_num_images: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes a list of list of images and stacks them into a tensor.
    This function is needed since images can be of completely
    different resolutions and aspect ratios.
    """
    out_images, out_num_chunks = [], []
    for imgs_sample in images:
        out_images_i = torch.zeros(
            max_num_images,
            max_num_chunks,
            3,
            image_res,
            image_res,
        )
        _num_chunks = []
        for j, chunks_image in enumerate(imgs_sample):
            out_images_i[j, : chunks_image.shape[0]] = chunks_image
            _num_chunks.append(chunks_image.shape[0])
        out_images.append(out_images_i)
        out_num_chunks.append(_num_chunks)
    return torch.stack(out_images), out_num_chunks


def _pad_masks(
    all_masks: List[List[List[int]]],  
    all_num_chunks: List[List[int]],
    total_len: int,
    max_num_chunks: int,
) -> torch.Tensor:
    """
    Example:
        Inputs: 
            all_masks = [
                [
                    [10, 20], # Mask for first media in the batch
                    [20, 21]  # Mask for second media in the batch
                ]
            ]
            all_num_chunks = [[2, 4]]
            total_len = 512
            max_num_chunks = 4
        Return: 
            A torch.Tensor out_masks of shape (1, 512, 2, 4) such that
                out_masks[0, 10:20, 0, :2] = 0.0
                out_masks[0, 20:21, 1, :4] = 0.0
                out_masks[otherwise] = -inf
    """
    dtype = torch.bfloat16
    inf_value = get_negative_inf_value(dtype)

    bsz = len(all_masks)
    max_num_media = max([len(m) for m in all_masks])

    out_masks = torch.full(
        (bsz, total_len, max_num_media, max_num_chunks),  # (1, 512, 2, 4)
        inf_value,
        dtype=dtype,
    )

    for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
        # idx=0, mask=[[10, 20], [20, 21]], num_chunks=[2, 4]
        for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
            # mask_idx=0, mask_elem=[10, 20], mask_num_chunks=2
            #   out_masks[0, 10:20, 0, :2] = 0.0
            # mask_idx=1, mask_elem=[20, 21], mask_num_chunks=4
            #   out_masks[0, 20:21, 1, :4] = 0.0
            if len(mask_elem) == 2:
                mask_elem[1] = min(mask_elem[1], total_len)
                if mask_elem[1] == -1:
                    mask_elem[1] = total_len
                out_masks[
                    idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks
                ].fill_(0.0)

    return out_masks
