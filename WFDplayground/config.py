from cvpods.configs.base_detection_config import BaseDetectionConfig
import math
from os.path import join, abspath, dirname, pardir

# Directories
BASE_DIR = abspath(join(dirname(__file__), pardir))
testGLUE = True

_config_dict = dict(
    MODEL=dict(
        LOAD_BACKBONE_WEIGHTS = True,#"True" for useing pretrained backbone model.
        WEIGHTS=join(BASE_DIR, "backbone_weights", "res18_1d.pth"), 
        # LOAD_BACKBONE_WEIGHTS = False,#"False" for useing full model checkpoint.
        # WEIGHTS=join(BASE_DIR, "WFDplayground/output", "model_final.pth"), 
        BACKBONE=dict(
            # Freeze parameters if FREEZE_AT >= 1
            FREEZE_AT=5,            
        ),
        RESNETS=dict(DEPTH=18, OUT_FEATURES=["res5"], NORM="BN1d",),
        
        ANCHOR_GENERATOR=dict(
            SIZES=[[71, 181, 279, 391, 496, 619, 771, 1007, 1282, 1784]] if testGLUE else
                [[196, 440, 682, 961, 1338, 1882, 2702, 3808, 5117, 6862]], 
            ASPECT_RATIOS=[[1.0]]
        ),
        WFD=dict(
            ENCODER=dict(
                IN_FEATURES=["res5"],
                NUM_CHANNELS=512,
                BLOCK_MID_CHANNELS=128,
                NUM_RESIDUAL_BLOCKS=4,
                BLOCK_DILATIONS=[2,4,6,8],
                NORM="BN1d",
                ACTIVATION="ReLU"
            ),
            DECODER=dict(
                IN_CHANNELS=512,
                NUM_CLASSES=200 if testGLUE else 100,
                NUM_ANCHORS=10,
                CLS_NUM_CONVS=2,
                REG_NUM_CONVS=4,
                NORM="BN1d",
                ACTIVATION="ReLU",
                PRIOR_PROB=0.01
            ),
            BBOX_REG_WEIGHTS=(1.0, 1.0),
            SCALE_CLAMP = math.log(1000.0 / 16),
            ADD_CTR_CLAMP=True,
            CTR_CLAMP=32,
            MATCHER_TOPK=4,
            POS_IGNORE_THRESHOLD=0.15,
            NEG_IGNORE_THRESHOLD=0.7,
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SCORE_THRESH_TEST=0.0005,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.6
        ),
    ),
    DATASETS=dict(
        TRAIN=("WFD_2022_train",),
        TEST=("WFD_2022_test",),
    ),
    DATALOADER=dict(NUM_WORKERS=1),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(8000 * 15, 9000 * 15),
            MAX_ITER=20,
            WARMUP_FACTOR=0.00066667 /4 * 1,
            WARMUP_ITERS=1500 * 4 / 1
        ),
        OPTIMIZER=dict(
            NAME="D2SGD",
            BASE_LR=0.012,
            BIAS_LR_FACTOR=1.0,
            WEIGHT_DECAY=0.0001,
            WEIGHT_DECAY_NORM=0.0,
            MOMENTUM=0.9,
            BACKBONE_LR_FACTOR=0.334
        ),
        CHECKPOINT_PERIOD=1000,
        IMS_PER_BATCH=64, 
        IMS_PER_DEVICE=64, 
    ),
    INPUT=dict(
        AUG=dict(

            TRAIN_PIPELINES=[
                ("PadTrace", dict(
                    top=0, left = 0, target_h = 1, target_w =10000,
                    pad_value=0)),
                ("RandomShiftTrace", dict(max_shifts=32)),
            ],
            TEST_PIPELINES=[
                # ("PadTrace", dict(
                #    top=0, left = 0, target_h = 1, target_w =10000,
                #     pad_value=0)),
            ],
        ),
        FORMAT="direction",
        start_iter =0,
        end_inter = 100000,
    ),
)


class WFDConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = WFDConfig()
