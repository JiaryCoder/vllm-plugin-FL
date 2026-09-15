# Copyright (c) 2026 BAAI. All rights reserved.
"""Functional torch rotary fallback for the FL normalized interface."""
import torch


def rotary_embedding_torch(
    obj, query: torch.Tensor, key: torch.Tensor | None,
    cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor,
    rotary_interleaved: bool = False, inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    cos_selected = cos[position_ids]
    sin_selected = sin[position_ids]
    if query.ndim == 4 and position_ids.ndim == 1:
        cos_selected = cos_selected.unsqueeze(0).unsqueeze(0)
        sin_selected = sin_selected.unsqueeze(0).unsqueeze(0)
    elif query.ndim in (3, 4):
        cos_selected = cos_selected.unsqueeze(1)
        sin_selected = sin_selected.unsqueeze(1)
    width = query.shape[-1]
    if cos_selected.shape[-1] * 2 == width:
        if rotary_interleaved:
            cos_selected = cos_selected.repeat_interleave(2, dim=-1)
            sin_selected = sin_selected.repeat_interleave(2, dim=-1)
        else:
            cos_selected = torch.cat((cos_selected, cos_selected), dim=-1)
            sin_selected = torch.cat((sin_selected, sin_selected), dim=-1)
    elif cos_selected.shape[-1] != width:
        raise ValueError("rotary cache width must be rotary_dim or rotary_dim / 2")

    def apply(x):
        if x is None:
            return None
        if rotary_interleaved:
            rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        else:
            x1, x2 = x.chunk(2, dim=-1)
            rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos_selected + rotated * sin_selected

    return apply(query), apply(key)
