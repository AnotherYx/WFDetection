# Copyright (c) WFDetection, Inc. and its affiliates. All Rights Reserved
import logging

import numpy as np

import torch
from torch import nn

from cvpods.layers import (
    Conv1d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_activation,
    get_norm
)
from cvpods.modeling.nn_utils import weight_init

from .backbone import Backbone

__all__ = [
    "ResNetBlockBase1d",
    "BottleneckBlock1d",
    "DeformBottleneckBlock1d",
    "BasicStem1d",
    "ResNet1d",
    "make_stage1d",
    "build_resnet1d_backbone",
]


class ResNetBlockBase1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (1,int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = (1,stride)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class BasicBlock1d(ResNetBlockBase1d):
    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN1d", activation=None,
                 **kwargs):
        """
        The standard block type for ResNet18 and ResNet34.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): A callable that takes the number of
                channels and returns a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN1d", "BN1d","GN"}).
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None
        self.activation = get_activation(activation)

        self.conv1 = Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)
        return out


class BottleneckBlock1d(ResNetBlockBase1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN1d",
        activation=None,
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN1d", "BN1d","GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.activation = get_activation(activation)

        self.conv1 = Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=(1 * dilation),
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv1d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)

        return out


class DeformBottleneckBlock1d(ResNetBlockBase1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN1d",
        activation=None,
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.activation = get_activation(activation)

        self.conv1 = Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv1d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=(1 * dilation),
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=(1 * dilation),
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv1d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.activation(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)
        return out


def make_stage1d(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.

    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class BasicStem1d(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN1d", activation=None,
                 deep_stem=False, stem_width=32):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN1d", "BN1d", "GN"}).
        """
        super().__init__()
        self.deep_stem = deep_stem

        if self.deep_stem:
            self.conv1_1 = Conv1d(
                3,
                stem_width,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=get_norm(norm, stem_width),
            )
            self.conv1_2 = Conv1d(
                stem_width,
                stem_width,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, stem_width),
            )
            self.conv1_3 = Conv1d(
                stem_width,
                stem_width * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, stem_width * 2),
            )
            for layer in [self.conv1_1, self.conv1_2, self.conv1_3]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        else:
            self.conv1 = Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
            weight_init.c2_msra_fill(self.conv1)

        self.activation = get_activation(activation)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.deep_stem:
            x = self.conv1_1(x)
            x = self.activation(x)
            x = self.conv1_2(x)
            x = self.activation(x)
            x = self.conv1_3(x)
            x = self.activation(x)
        else:
            x = self.conv1(x)
            x = self.activation(x)
        x = self.max_pool(x)
        return x

    @property
    def out_channels(self):
        if self.deep_stem:
            return self.conv1_3.out_channels
        else:
            return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class ResNet1d(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None, zero_init_residual=False):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet1d, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase1d), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.linear = nn.Linear(curr_channels, num_classes)

            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock1d):
                    nn.init.constant_(m.conv3.norm.weight, 0)
                elif isinstance(m, BasicBlock1d):
                    nn.init.constant_(m.conv2.norm.weight, 0)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def build_resnet1d_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    depth = cfg.MODEL.RESNETS.DEPTH
    stem_width = {18: 32, 34: 32, 50: 32, 101: 64, 152: 64, 200: 64, 269: 64}[depth]
    deep_stem = cfg.MODEL.RESNETS.DEEP_STEM

    if not deep_stem:
        assert getattr(cfg.MODEL.RESNETS, "RADIX", 1) <= 1, \
            "cfg.MODEL.RESNETS.RADIX: {} > 1".format(cfg.MODEL.RESNETS.RADIX)

    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    activation = cfg.MODEL.RESNETS.ACTIVATION
    stem = BasicStem1d(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        activation=activation,
        deep_stem=deep_stem,
        stem_width=stem_width,
    )
    # freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    freeze_at = 0

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    num_classes = cfg.MODEL.RESNETS.NUM_CLASSES
    zero_init_residual = cfg.MODEL.RESNETS.ZERO_INIT_RESIDUAL
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
        269: [3, 30, 48, 8],
    }[depth]

    # Avoid creating variables without gradients
    # which consume extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    deform_on_per_stage = getattr(cfg.MODEL.RESNETS,
                                  "DEFORM_ON_PER_STAGE",
                                  [False] * (max_stage_idx - 1))
                                  
    '''[WFD 2022]'''
    if depth in [18, 34]:
        cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        cfg.MODEL.RESNETS.RES5_DILATION = 1
        res5_dilation = 1

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    in_channels = 2 * stem_width if deep_stem else in_channels
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            "activation": activation,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock1d
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock1d
                # Use True to use modulated deform_conv (DeformableV2);
                # Use False for DeformableV1.
                stage_kargs["deform_modulated"] = cfg.MODEL.RESNETS.DEFORM_MODULATED
                # Number of groups in deformable conv.
                stage_kargs["deform_num_groups"] = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
            else:
                stage_kargs["block_class"] = BottleneckBlock1d
        blocks = make_stage1d(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)

    return ResNet1d(stem,
                  stages,
                  num_classes=num_classes,
                  out_features=out_features,
                  zero_init_residual=zero_init_residual)
