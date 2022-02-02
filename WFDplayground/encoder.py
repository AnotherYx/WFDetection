from typing import List
from easydict import EasyDict as edict

import torch
import torch.nn as nn

from cvpods.layers import ShapeSpec, get_norm, get_activation
from cvpods.modeling.nn_utils import weight_init

class Encoder(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(Encoder, self).__init__()
        # fmt: off
        self.backbone_level = cfg.MODEL.WFD.ENCODER.IN_FEATURES
        self.in_channels = input_shape[self.backbone_level[0]].channels
        self.encoder_channels = cfg.MODEL.WFD.ENCODER.NUM_CHANNELS
        self.block_mid_channels = cfg.MODEL.WFD.ENCODER.BLOCK_MID_CHANNELS
        self.num_residual_blocks = cfg.MODEL.WFD.ENCODER.NUM_RESIDUAL_BLOCKS
        self.block_dilations = cfg.MODEL.WFD.ENCODER.BLOCK_DILATIONS
        self.norm_type = cfg.MODEL.WFD.ENCODER.NORM
        self.act_type = cfg.MODEL.WFD.ENCODER.ACTIVATION
        # fmt: on
        assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv1d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=1)
        self.lateral_norm = get_norm(self.norm_type, self.encoder_channels)
        self.fpn_conv = nn.Conv1d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = get_norm(self.norm_type, self.encoder_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation,
                    norm_type=self.norm_type,
                    act_type=self.act_type
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weight(self):
        weight_init.c2_xavier_fill(self.lateral_conv)
        weight_init.c2_xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1,
                 norm_type: str = 'BN1d',
                 act_type: str = 'ReLU'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, padding=0),
            get_norm(norm_type, mid_channels),
            get_activation(edict({'NAME': act_type, 'INPLACE': True}))
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            get_norm(norm_type, mid_channels),
            get_activation(edict({'NAME': act_type, 'INPLACE': True}))
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(mid_channels, in_channels, kernel_size=1, padding=0),
            get_norm(norm_type, in_channels),
            get_activation(edict({'NAME': act_type, 'INPLACE': True}))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


def build_encoder(cfg, input_shape: ShapeSpec):
    return Encoder(cfg, input_shape=input_shape)
