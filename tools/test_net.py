# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved
# Modified by WFDetection, Inc. and its affiliates. All Rights Reserved
import glob
import os
import re
from pprint import pformat

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.engine import RUNNERS, default_argument_parser, default_setup, launch
from cvpods.evaluation import build_evaluator, verify_results
from cvpods.utils import PathManager, comm
from config import config
from net import build_model


def runner_decrator(cls):

    def custom_build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        dump_test = config.GLOBAL.DUMP_TEST
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_test)
    cls.build_evaluator = classmethod(custom_build_evaluator)

    return cls


def test_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--start-iter", type=int, default=None, help="start iter used to test")
    parser.add_argument("--end-iter", type=int, default=None, help="end iter used to test")
    parser.add_argument("--debug", action="store_true", help="use debug mode or not")
    return parser


def filter_by_iters(file_list, start_iter, end_iter):
    # sort file_list by modified time
    if file_list[0].startswith("s3://"):
        file_list.sort(key=lambda x: PathManager.stat(x).m_date)
    else:
        file_list.sort(key=os.path.getmtime)

    if start_iter is None:
        if end_iter is None:
            # use latest ckpt if start_iter and end_iter are not given
            return [file_list[-1]]
        else:
            start_iter = 0
    elif end_iter is None:
        end_iter = float("inf")

    iter_infos = [re.split(r"model_|\.pth", f)[-2] for f in file_list]
    # My add
    iter_infos = [int(re.split(r"_", f)[-1]) for f in iter_infos if "iter" in f]
    iter_infos.append("final")
    keep_list = [0] * len(iter_infos)
    start_index = 0
    if "final" in iter_infos and iter_infos[-1] != "final":
        start_index = iter_infos.index("final")

    for i in range(len(iter_infos) - 1, start_index, -1):
        if iter_infos[i] == "final":
            if end_iter == float("inf"):
                keep_list[i] = 1
        elif float(start_iter) <= float(iter_infos[i]) < float(end_iter):
            keep_list[i] = 1
            if float(iter_infos[i - 1]) > float(iter_infos[i]):
                break
    keep_list[0] = 1 # my add            
    return [filename for keep, filename in zip(keep_list, file_list) if keep == 1]


def get_valid_files(args, cfg, logger):
    if "MODEL.WEIGHTS" in args.opts:
    # if "WEIGHTS" in cfg.MODEL:
        model_weights = cfg.MODEL.WEIGHTS
        assert PathManager.exists(model_weights), "{} not exist!!!".format(model_weights)
        return [model_weights]

    file_list = glob.glob(os.path.join(cfg.OUTPUT_DIR, "model_*.pth"))
    print(cfg.OUTPUT_DIR, "cfg.OUTPUT_DIR")
    if len(file_list) == 0:  # local file invalid, get it from oss
        model_prefix = cfg.OUTPUT_DIR.split("cvpods_playground")[-1][1:]
        remote_file_path = os.path.join(cfg.OSS.DUMP_PREFIX, model_prefix)
        logger.warning(
            "No checkpoint file was found locally, try to "
            f"load the corresponding dump file on OSS site: {remote_file_path}."
        )
        file_list = [
            str(filename) for filename in PathManager.ls(remote_file_path)
            if re.match(r"model_.+\.pth", filename.name) is not None
        ]
        assert len(file_list) != 0, "No valid file found on OSS"
    args.start_iter = 0
    args.end_iter = None
    file_list = filter_by_iters(file_list, args.start_iter, args.end_iter)
    file_list.reverse()
    file_list = file_list[0:1]
    assert file_list, "No checkpoint valid in {}.".format(cfg.OUTPUT_DIR)
    logger.info("All files below will be tested in order:\n{}".format(pformat(file_list)))
    return file_list


def main(args):
    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)
    if args.debug:
        batches = int(cfg.SOLVER.IMS_PER_DEVICE * args.num_gpus)
        if cfg.SOLVER.IMS_PER_BATCH != batches:
            cfg.SOLVER.IMS_PER_BATCH = batches
            logger.warning("SOLVER.IMS_PER_BATCH is changed to {}".format(batches))

    valid_files = get_valid_files(args, cfg, logger)
    # * means all if need specific format then *.csv
    for current_file in valid_files:
        cfg.MODEL.WEIGHTS = current_file
        model = build_model(cfg)

        DefaultCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = runner_decrator(RUNNERS.get(cfg.TRAINER.NAME)).test(cfg, model)

        if comm.is_main_process():
            verify_results(cfg, res)

if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    # print("Command Line Args:", args) 
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
