# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by WFDetection, Inc. and its affiliates. All Rights Reserved
import math
from enum import IntEnum, unique
from typing import Iterator, List, Tuple, Union

import numpy as np

import torch
from torchvision.ops.boxes import box_area

from cvpods.layers import cat

# _RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]

_RawIntervalType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]
# @unique
# class BoxMode(IntEnum):
#     """
#     Enum of different ways to represent a box.

#     Attributes:

#         XYXY_ABS: (x0, y0, x1, y1) in absolute floating points coordinates.
#             The coordinates in range [0, width or height].
#         XYWH_ABS: (x0, y0, w, h) in absolute floating points coordinates.
#         XYXY_REL: (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
#         XYWH_REL: (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
#         XYWHA_ABS: (xc, yc, w, h, a) in absolute floating points coordinates.
#             (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
#     """

#     XYXY_ABS = 0
#     XYWH_ABS = 1
#     XYXY_REL = 2
#     XYWH_REL = 3
#     XYWHA_ABS = 4

#     @staticmethod
#     def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
#         """
#         Args:
#             box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
#             from_mode, to_mode (BoxMode)

#         Returns:
#             The converted box of the same type.
#         """
#         if from_mode == to_mode:
#             return box

#         original_type = type(box)
#         is_numpy = isinstance(box, np.ndarray)
#         single_box = isinstance(box, (list, tuple))
#         if single_box:
#             assert len(box) == 4 or len(box) == 5, (
#                 "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
#                 " where k == 4 or 5"
#             )
#             arr = torch.tensor(box)[None, :]
#         else:
#             # avoid modifying the input box
#             if is_numpy:
#                 arr = torch.from_numpy(np.asarray(box)).clone()
#             else:
#                 arr = box.clone()

#         assert to_mode.value not in [
#             BoxMode.XYXY_REL,
#             BoxMode.XYWH_REL,
#         ] and from_mode.value not in [
#             BoxMode.XYXY_REL,
#             BoxMode.XYWH_REL,
#         ], "Relative mode not yet supported!"

#         if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
#             assert (
#                 arr.shape[-1] == 5
#             ), "The last dimension of input shape must be 5 for XYWHA format"

#             original_dtype = arr.dtype
#             arr = arr.double()

#             w = arr[:, 2]
#             h = arr[:, 3]
#             a = arr[:, 4]
#             c = torch.abs(torch.cos(a * math.pi / 180.0))
#             s = torch.abs(torch.sin(a * math.pi / 180.0))
#             # This basically computes the horizontal bounding rectangle of the rotated box
#             new_w = c * w + s * h
#             new_h = c * h + s * w

#             # convert center to top-left corner
#             arr[:, 0] -= new_w / 2.0
#             arr[:, 1] -= new_h / 2.0
#             # bottom-right corner
#             arr[:, 2] = arr[:, 0] + new_w
#             arr[:, 3] = arr[:, 1] + new_h

#             arr = arr[:, :4].to(dtype=original_dtype)
#         elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
#             original_dtype = arr.dtype
#             arr = arr.double()
#             arr[:, 0] += arr[:, 2] / 2.0
#             arr[:, 1] += arr[:, 3] / 2.0
#             angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
#             arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
#         else:
#             if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
#                 arr[:, 2] += arr[:, 0]
#                 arr[:, 3] += arr[:, 1]
#             elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
#                 arr[:, 2] -= arr[:, 0]
#                 arr[:, 3] -= arr[:, 1]
#             else:
#                 raise NotImplementedError(
#                     "Conversion from BoxMode {} to {} is not supported yet".format(
#                         from_mode, to_mode
#                     )
#                 )

#         if single_box:
#             return original_type(arr.flatten())

#         if is_numpy:
#             return arr.numpy()
#         else:
#             return arr
@unique
class IntervalMode(IntEnum):
    """
    Enum of different ways to represent a Interval.

    Attributes:

        XX_ABS: (x0, x1) in absolute floating points coordinates.
            The coordinates in range [0, length].
        XW_ABS: (x0, w) in absolute floating points coordinates.
    """

    XX_ABS = 0
    XW_ABS = 1
    @staticmethod
    def convert(interval: _RawIntervalType, from_mode: "IntervalMode", to_mode: "IntervalMode") -> _RawIntervalType:
        """
        Args:
            Interval: can be a k-tuple, k-list or an Nxk array/tensor, where k = 2
            from_mode, to_mode (IntervalMode)

        Returns:
            The converted interval of the same type.
        """
        if from_mode == to_mode:
            return interval

        original_type = type(interval)
        is_numpy = isinstance(interval, np.ndarray)
        single_interval = isinstance(interval, (list, tuple))
        if single_interval:
            assert len(interval) == 2, (
                "IntervalMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 2"
            )
            arr = torch.tensor(interval)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(interval)).clone()
            else:
                arr = interval.clone()

        # assert to_mode.value not in [
        #     IntervalMode.XX_ABS,
        #     IntervalMode.XW_ABS,
        # ] and from_mode.value not in [
        #     IntervalMode.XX_ABS,
        #     IntervalMode.XW_ABS,
        # ], "Relative mode not yet supported!"

        # if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
        #     assert (
        #         arr.shape[-1] == 5
        #     ), "The last dimension of input shape must be 5 for XYWHA format"

        #     original_dtype = arr.dtype
        #     arr = arr.double()

        #     w = arr[:, 2]
        #     h = arr[:, 3]
        #     a = arr[:, 4]
        #     c = torch.abs(torch.cos(a * math.pi / 180.0))
        #     s = torch.abs(torch.sin(a * math.pi / 180.0))
        #     # This basically computes the horizontal bounding rectangle of the rotated box
        #     new_w = c * w + s * h
        #     new_h = c * h + s * w

        #     # convert center to top-left corner
        #     arr[:, 0] -= new_w / 2.0
        #     arr[:, 1] -= new_h / 2.0
        #     # bottom-right corner
        #     arr[:, 2] = arr[:, 0] + new_w
        #     arr[:, 3] = arr[:, 1] + new_h

        #     arr = arr[:, :4].to(dtype=original_dtype)
        # elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
        #     original_dtype = arr.dtype
        #     arr = arr.double()
        #     arr[:, 0] += arr[:, 2] / 2.0
        #     arr[:, 1] += arr[:, 3] / 2.0
        #     angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
        #     arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        # else:
        if to_mode == IntervalMode.XX_ABS and from_mode == IntervalMode.XW_ABS:
            # center_position = arr[:, 0]
            # length = arr[:, 1]
            arr[:, 0] = arr[:, 0] - arr[:, 1] / 2 #start_pos
            arr[:, 1] = arr[:, 0] + arr[:, 1]  #end_pos
        elif from_mode == IntervalMode.XX_ABS and to_mode == IntervalMode.XW_ABS:
            arr[:, 1] -= arr[:, 0] # length
            arr[:, 0] = arr[:, 0] + arr[:, 1] / 2 # center_pos
        else:
            raise NotImplementedError(
                "Conversion from BoxMode {} to {} is not supported yet".format(
                    from_mode, to_mode
                )
            )

        if single_interval:
            return original_type(arr.flatten())

        if is_numpy:
            return arr.numpy()
        else:
            return arr

class Intervales:
    """
    This structure stores a list of intervales as a Nx2 torch.Tensor.
    It supports some common methods about intervales
    (`length`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all intervales)

    Attributes:
        tensor(torch.Tensor): float matrix of Nx2.
    """

    # IntervalSizeType = Union[List[int], Tuple[int, int]]
    IntervalSizeType = int
    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx2 matrix.  Each row is (x1, x2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 4, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 2, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Intervales":
        """
        Clone the Intervales.

        Returns:
            Intervales
        """
        return Intervales(self.tensor.clone())

    def to(self, device: str) -> "Intervales":
        return Intervales(self.tensor.to(device))

    def length(self) -> torch.Tensor:
        """
        Computes the length of all the intervales.

        Returns:
            torch.Tensor: a vector with lengths of each interval.
        """
        interval = self.tensor
        length = interval[:, 1] - interval[:, 0]
        return length

    def clip(self, interval_size: IntervalSizeType) -> None:
        """
        Clip (in place) the intervales by limiting x coordinates to the range [0, length]
        and y coordinates to the range [0, length].

        Args:
            interval_size (): The clipping interval's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        w = interval_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=w)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        Find intervales that are non-empty.
        A interval is considered empty, if its length is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each interval is empty
                (False) or non-empty (True).
        """
        interval = self.tensor
        lengths = interval[:, 1] - interval[:, 0]
        keep = lengths > threshold
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Intervales":
        """
        Returns:
            Intervales: Create a new :class:`Intervales` by indexing.

        The following usage are allowed:

        1. `new_intervales = intervales[3]`: return a `intervales` which contains only one interval.
        2. `new_intervales = intervales[2:10]`: return a slice of intervales.
        3. `new_intervales = intervales[vector]`, where vector is a torch.BoolTensor
        with `length = len(intervales)`. Nonzero elements in the vector will be selected.

        Note that the returned Intervales might share storage with this Intervales,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Intervales(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Intervales(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Intervales(" + str(self.tensor) + ")"

    def inside_Interval(self, interval_size: IntervalSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            interval_size (length): Size of the reference interval.
            boundary_threshold (int): Intervales that extend beyond the reference interval
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each interval is inside the reference interval.
        """
        length = interval_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] < length + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The interval centers in a Nx1 array of (x).
        """
        return (self.tensor[:, 1] + self.tensor[:, 0]) / 2

    def scale(self, scale: float) -> None:
        """
        Scale the interval with scaling factors
        """
        self.tensor[:, :] *= scale
        # self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, intervales_list: List["Intervales"]) -> "Intervales":
        """
        Concatenates a list of Intervales into a single Intervales

        Arguments:
            intervales_list (list[Intervales])

        Returns:
            Intervales: the concatenated Boxes
        """
        assert isinstance(intervales_list, (list, tuple))
        assert all(isinstance(interval, Intervales) for interval in intervales_list)

        if len(intervales_list) == 0:
            return cls(torch.empty(0))

        cat_intervales = type(intervales_list[0])(cat([b.tensor for b in intervales_list], dim=0))
        return cat_intervales

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a interval as a Tensor of shape (2,) at a time.
        """
        yield from self.tensor


# added for DETR
# TODO @wangfeng02, use BoxMode instead and provide a better func
# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)
def interval_cxw_to_xx(x):
    x_c, w= x.unbind(-1)
    b = [(x_c - 0.5 * w), (x_c + 0.5 * w)]
    return torch.stack(b, dim=-1)

# def box_xyxy_to_cxcywh(x):
#     x0, y0, x1, y1 = x.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)
def interval_xx_to_cxcyw(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (x1 - x0)]
    return torch.stack(b, dim=-1)

def interval_length(intervales):
    """
    Computes the area of a set of bounding intervales, which are specified by its
    (x1,x2) coordinates.

    Arguments:
        intervales (Tensor[N, 2]): intervales for which the area will be computed. They
            are expected to be in (x1, x2) format

    Returns:
        length (Tensor[N]): length for each interval
    """
    return intervales[:, 1] - intervales[:, 0]

def generalized_interval_iou_trace(intervales1, intervales2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The intervales should be in [x0, x1] format
    Returns a [N, M] pairwise matrix, where N = len(intervales1)
    and M = len(intervales2)
    """
    # degenerate intervales gives inf / nan results
    # so do an early check
    assert (intervales1[:, 1] >= intervales1[:, 0]).all()
    assert (intervales2[:, 1] >= intervales2[:, 0]).all()

    # vallina interval iou
    # modified from torchvision to also return the union
    length1 = interval_length(intervales1)
    length2 = interval_area(intervales2)

    l = torch.max(intervales1[:, None, 0], intervales2[:, 0])  # [N,M]
    r = torch.min(intervales1[:, None, 1], intervales2[:, 1])  # [N,M]

    w = (r - l).clamp(min=0)  # [N,M]
    inter = w  # [N,M]

    union = length1[:, None] + length2 - inter
    iou = inter / union

    # iou, union = box_iou(boxes1, boxes2)

    l2 = torch.min(intervales1[:, None, 0], intervales2[:, 0])
    r2 = torch.max(intervales1[:, None, 1], intervales2[:, 1])

    w2 = (r2 - l2).clamp(min=0)  # [N,M]
    # area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (w2 - union) / w2


def masks_to_boxes(masks):
    """
    Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks,
    (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
