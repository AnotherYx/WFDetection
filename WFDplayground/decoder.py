import math
from typing import Tuple
from easydict import EasyDict as edict

import torch
import torch.nn as nn
from cvpods.layers import get_norm,get_activation

class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        # fmt: off
        self.in_channels = cfg.MODEL.WFD.DECODER.IN_CHANNELS
        self.num_classes = cfg.MODEL.WFD.DECODER.NUM_CLASSES
        self.num_anchors = cfg.MODEL.WFD.DECODER.NUM_ANCHORS
        self.cls_num_convs = cfg.MODEL.WFD.DECODER.CLS_NUM_CONVS
        self.reg_num_convs = cfg.MODEL.WFD.DECODER.REG_NUM_CONVS
        self.norm_type = cfg.MODEL.WFD.DECODER.NORM
        self.act_type = cfg.MODEL.WFD.DECODER.ACTIVATION
        self.prior_prob = cfg.MODEL.WFD.DECODER.PRIOR_PROB
        # fmt: on

        self.INF = 1e8
        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.cls_num_convs):
            cls_subnet.append(
                nn.Conv1d(self.in_channels,
                          self.in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(get_norm(self.norm_type, self.in_channels))
            cls_subnet.append(get_activation(edict({'NAME': self.act_type, 'INPLACE': True})))
        for i in range(self.reg_num_convs):
            bbox_subnet.append(
                nn.Conv1d(self.in_channels,
                          self.in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(get_norm(self.norm_type, self.in_channels))
            bbox_subnet.append(get_activation(edict({'NAME': self.act_type, 'INPLACE': True})))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv1d(self.in_channels,
                                   self.num_anchors * self.num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv1d(self.in_channels,
                                   self.num_anchors * 2,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.object_pred = nn.Conv1d(self.in_channels,
                                     self.num_anchors,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self,
                feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_score = self.cls_score(self.cls_subnet(feature))

        N, _, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=self.INF) + torch.clamp(
                objectness.exp(), max=self.INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, W)
        return normalized_cls_score, bbox_reg


def build_decoder(cfg):
    return Decoder(cfg)



