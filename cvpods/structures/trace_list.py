# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by WFDetection, Inc. and its affiliates. All Rights Reserved
from __future__ import division
from typing import Any, List, Sequence, Tuple, Union

import torch
from torch.nn import functional as F


class TraceList(object):
    """
    Structure that holds a list of traces (of possibly
    varying sizes) as a single tensor.
    This works by padding the traces to the same size,
    and storing in a field the original sizes of each trace

    Attributes:
        trace_sizes (list[tuple[int, int]]): each tuple is (h, w)
    """

    def __init__(self, tensor: torch.Tensor, trace_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            trace_sizes (list[tuple[int, int]]): Each tuple is (h, w).
        """
        self.tensor = tensor
        self.trace_sizes = trace_sizes

    def __len__(self) -> int:
        return len(self.trace_sizes)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        """
        Access the individual trace in its original size.

        Returns:
            Tensor: an trace of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.trace_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]  # type: ignore

    def to(self, *args: Any, **kwargs: Any) -> "TraceList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return TraceList(cast_tensor, self.trace_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: Sequence[torch.Tensor],
        size_divisibility: int = 0,
        pad_ref_long: bool = False,
        pad_value: float = 0.0,
    ) -> "TraceList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded with `pad_value`
                so that they will have the same shape.
            size_divisibility (int): If `size_divisibility > 0`, also adds padding to ensure
                the common height and width is divisible by `size_divisibility`
            pad_value (float): value to pad

        Returns:
            an `TraceList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
        max_size = list(max(s) for s in zip(*[trc.shape for trc in tensors]))
        if pad_ref_long:
            max_size_max = max(max_size[-2:])
            max_size[-2:] = [max_size_max] * 2
        max_size = tuple(max_size)

        if size_divisibility > 0:
            import math

            stride = size_divisibility
            max_size = list(max_size)  # type: ignore
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)  # type: ignore
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)  # type: ignore
            max_size = tuple(max_size)

        trace_sizes = [im.shape[-2:] for im in tensors]

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple traces as well
            trace_size = trace_sizes[0]
            padding_size = [0, max_size[-1] - trace_size[1], 0, max_size[-2] - trace_size[0]]
            if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
                batched_trcs = tensors[0].unsqueeze(0)
            else:
                padded = F.pad(tensors[0], padding_size, value=pad_value)
                batched_trcs = padded.unsqueeze_(0)
        else:
            batch_shape = (len(tensors),) + max_size
            batched_trcs = tensors[0].new_full(batch_shape, pad_value)
            for trc, pad_trc in zip(tensors, batched_trcs):
                pad_trc[..., : trc.shape[-2], : trc.shape[-1]].copy_(trc)
        batched_trcs = batched_trcs.squeeze(dim=1)
        return TraceList(batched_trcs.contiguous(), trace_sizes)
