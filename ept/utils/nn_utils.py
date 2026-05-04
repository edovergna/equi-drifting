#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_sum


def std_conserve_scatter_sum(src, index, dim):
    ones = torch.ones_like(index)
    n = scatter_sum(ones, index, dim=0)  # [N]
    value = scatter_sum(src, index, dim=dim)  # [N, ...]
    value = value / torch.sqrt(n).unsqueeze(-1)
    return value


def stable_norm(input, *args, **kwargs):
    input = input.clone()
    with torch.no_grad():
        sign = torch.sign(input)
        input = torch.abs(input)
        input.clamp_(min=1e-10)
        input = sign * input
    return torch.norm(input, *args, **kwargs)


def graph_to_batch_nx(
    tensor, batch_id, padding_value=0, mask_is_pad=True, factor_req=8
):
    """
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    """
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    max_n = max_n if (max_n % 8 == 0) else (max_n // 8 * 8 + 8)
    batch = (
        torch.ones(
            (bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        * padding_value
    )
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask
