from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone import build_resnet1d_backbone
from cvpods.modeling.anchor_generator import WFDAnchorGenerator

from encoder import build_encoder
from decoder import build_decoder
from wfdetection import WFDetection


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=1)

    backbone = build_resnet1d_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_anchor_generator(cfg, input_shape):
    return WFDAnchorGenerator(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_anchor_generator = build_anchor_generator
    cfg.build_encoder = build_encoder
    cfg.build_decoder = build_decoder
    model = WFDetection(cfg)

    return model
