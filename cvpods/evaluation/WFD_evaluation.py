# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
from pycocotools.coco import COCO

import torch

from cvpods.data.datasets.coco import convert_to_coco_json
from cvpods.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from cvpods.structures import BoxMode
from cvpods.utils import PathManager, comm, create_small_table, create_table_with_header

from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class WFDEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, meta, cfg, distributed, output_dir=None, dump=False):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in cvpods's standard dataset format
                so it can be converted to COCO format automatically.
            meta (SimpleNamespace): dataset metadata.
            cfg (config dict): cvpods Config instance.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.

            dump (bool): If True, after the evaluation is completed, a Markdown file
                that records the model evaluation metrics and corresponding scores
                will be generated in the working directory.
        """
        self._dump = dump
        self.cfg = cfg
        self._tasks = ("bbox",)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = meta
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
                " Trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []
        self._dump_infos = []  # per task

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "trace_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"trace_id": input["trace_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["trace_id"])

            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions(set(self._tasks))

        if self._dump:
            extra_infos = {
                "title": os.path.basename(os.getcwd()),
                "seed": self.cfg.SEED,
            }
            _dump_to_markdown(extra_infos, self._dump_infos)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if len(reverse_id_mapping) == 200:
            glue=True
        else:
            glue=False
        for task in sorted(tasks):
            coco_eval, summary = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, task, glue, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            self._logger.info("\n" + summary.getvalue())
            res = self._derive_coco_results(
                coco_eval, task, summary, class_names=self._metadata.thing_classes
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, summary, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str): specific evaluation task,
                optional values are: "bbox", "segm", "keypoints".
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        small_table = create_small_table(results)
        self._logger.info("Evaluation results for {}: \n".format(iou_type) + small_table)
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None:  # or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = {}
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category[name] = float(ap * 100)
            # results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        table = create_table_with_header(results_per_category, headers=["category", "AP"])
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + str(name): ap for name, ap in results_per_category.items()})
        if self._dump:
            dump_info_one_task = {
                "task": iou_type,
                "summary": summary.getvalue(),
                "tables": [small_table, table],
            }
            self._dump_infos.append(dump_info_one_task)
        return results


def instances_to_coco_json(instances, trc_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        trc_id (int): the trace id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    # boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XCYCWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "trace_id": trc_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }

        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, glue=False, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.areaRngLbl = ['all', 'short', 'medium', 'long']
    if glue:
        coco_eval.params.areaRng  = [[0, 4000], [0, 128], [128, 1000], [1000, 4000]] 
    else:
        coco_eval.params.areaRng  = [[0, 9000], [0, 1000], [1000, 4000], [4000, 9000]] 
    
    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

    coco_eval.evaluate()
    coco_eval.accumulate()

    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_eval.summarize()
    redirect_string.getvalue()
    return coco_eval, redirect_string


def _dump_to_markdown(extra_infos, dump_infos, md_file="README.md"):
    """
    Dump a Markdown file that records the model evaluation metrics and corresponding scores
    to the current working directory.

    Args:
        extra_infos (dict): extra_infos to dump.
        dump_infos (list[dict]): dump information for each task.
        md_file (str): markdown file path.
    """
    with open(md_file, "w") as f:
        title = extra_infos["title"]
        seed = extra_infos["seed"]
        f.write("# {}  \n\nseed: {}".format(title, seed))
        for task_info in dump_infos:
            task_name = task_info["task"]
            f.write("\n\n## Evaluation results for {}:  \n\n".format(task_name))
            f.write("```  \n" + task_info["summary"] + "```  \n")
            overall_table, detail_table = [
                table.replace("\n", "  \n") for table in task_info["tables"]
            ]
            header = "\n\n### Per-category {} AP:  \n\n".format(task_name)
            f.write(overall_table + header + detail_table + "\n")
