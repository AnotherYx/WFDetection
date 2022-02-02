# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by BaseDetection, Inc. and its affiliates. All Rights Reserved
# Modified by WFDetection, Inc. and its affiliates. All Rights Reserved
import os
from colorama import Fore, Style

from cvpods.engine import RUNNERS, default_argument_parser, default_setup, launch
from cvpods.evaluation import build_evaluator
from config import config
from net import build_model


def runner_decrator(cls):

    def custom_build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        dump_train = config.GLOBAL.DUMP_TRAIN
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_train)

    cls.build_evaluator = classmethod(custom_build_evaluator)

    return cls


def main(args):
    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)

    runner = runner_decrator(RUNNERS.get(cfg.TRAINER.NAME))(cfg, build_model)
    runner.resume_or_load(resume=args.resume)

    # check wheather worksapce has enough storeage space
    # assume that a single dumped model is 700Mb
    file_sys = os.statvfs(cfg.OUTPUT_DIR)
    free_space_Gb = (file_sys.f_bfree * file_sys.f_frsize) / 2**30
    eval_space_Gb = (cfg.SOLVER.LR_SCHEDULER.MAX_ITER // cfg.SOLVER.CHECKPOINT_PERIOD) * 700 / 2**10
    if eval_space_Gb > free_space_Gb:
        logger.warning(f"{Fore.RED}Remaining space({free_space_Gb}GB) "
                       f"is less than ({eval_space_Gb}GB){Style.RESET_ALL}")

    # logger.info("Running with full config:\n{}".format(cfg))
    # base_config = cfg.__class__.__base__()
    # logger.info("different config with base class:\n{}".format(cfg.diff(base_config))) 

    runner.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    config.link_log()
    # print("soft link to {}".format(config.OUTPUT_DIR))
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
