#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.
import contextlib
import copy
import datetime
import io
import json
import logging
import os
import os.path as osp

import numpy as np
from PIL import Image

import torch

from cvpods.structures import BoxMode
from cvpods.utils import PathManager, Timer

from ..base_dataset import BaseDataset
from ..detection_utils import (
    annotations_to_instances,
    check_trace_size,
    filter_empty_instances,
    load_trace
)
from ..registry import DATASETS
from .builtin_meta import _get_builtin_metadata
from .paths_route import _PREDEFINED_SPLITS_WFD
from pycocotools.coco import COCO
"""
This file contains functions to parse COCO-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger(__name__)


@DATASETS.register()
class WFDDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(WFDDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        self.task_key = "WFD"          # for task: instance detection/segmentation
        self.meta = self._get_metadata()
        self.dataset_dicts = self._load_annotations(
            self.meta["json_file"],
            self.meta["trace_root"],
            dataset_name)

        # fmt: off
        self.data_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
        # fmt: on

        if is_train:
            # Remove traces without instance-level GT even though the dataset has semantic labels.
            self.dataset_dicts = self._filter_annotations(
                filter_empty=self.filter_empty,
                min_keypoints=0,
                proposal_files=None,
            )
            self._set_group_flag()

        # self.eval_with_gt = cfg.TEST.get("WITH_GT", False)
        self.eval_with_gt = cfg.TEST.get("WITH_GT", True)
        self.keypoint_hflip_indices = None

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read trace
        trace = load_trace(dataset_dict["file_name"], format=self.data_format)
        
        check_trace_size(dataset_dict, trace)

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [
                ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        trace = trace[np.newaxis, ..., np.newaxis]
        # apply transfrom
        trace, annotations = self._apply_transforms(
            trace, annotations, keypoint_hflip_indices=self.keypoint_hflip_indices)

        if annotations is not None:  # got instances in annotations
            trace_shape = trace.shape[:2]  # h, w
            instances = annotations_to_instances(
                annotations, trace_shape, mask_format=self.mask_format
            )
            dataset_dict["instances"] = filter_empty_instances(instances)
        dataset_dict["trace"] = torch.as_tensor(
            np.ascontiguousarray(trace.transpose(2, 0, 1)), dtype=torch.float)

        return dataset_dict

    def __reset__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_dicts)

    def _load_annotations(self,
                          json_file,
                          trace_root,
                          dataset_name=None,
                          extra_annotation_keys=None):

        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds()))

        id_map = None
        if dataset_name is not None:
            cat_ids = sorted(coco_api.getCatIds())
            cats = coco_api.loadCats(cat_ids)
            # The categories in a custom json file may not be sorted.
            thing_classes = [
                c["name"] for c in sorted(cats, key=lambda x: x["id"])
            ]
            self.meta["thing_classes"] = thing_classes

            if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
                if "WFD" not in dataset_name:
                    logger.warning("""
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
    """)
            id_map = {v: i for i, v in enumerate(cat_ids)}
            self.meta["thing_dataset_id_to_contiguous_id"] = id_map

        # sort indices for reproducible results
        trc_ids = sorted(coco_api.imgs.keys())

        trcs = coco_api.loadImgs(trc_ids)

        anns = [coco_api.imgToAnns[trc_id] for trc_id in trc_ids]

        ann_ids = [
            ann["id"] for anns_per_trace in anns for ann in anns_per_trace
        ]
        assert len(set(ann_ids)) == len(
            ann_ids), "Annotation ids in '{}' are not unique!".format(
                json_file)

        trcs_anns = list(zip(trcs, anns))

        logger.info("Loaded {} traces in COCO format from {}".format(
            len(trcs_anns), json_file))

        dataset_dicts = []

        ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"
                    ] + (extra_annotation_keys or [])

        num_instances_without_valid_segmentation = 0

        for (trc_dict, anno_dict_list) in trcs_anns:
            record = {}
            record["file_name"] = os.path.join(trace_root,
                                               trc_dict["file_name"])
            record["height"] = trc_dict["height"]
            record["width"] = trc_dict["width"]
            trace_id = record["trace_id"] = trc_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["trace_id"] == trace_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".
                format(num_instances_without_valid_segmentation))
        return dataset_dicts

    def _get_metadata(self):
        meta = _get_builtin_metadata(self.task_key)
        trace_root, json_file = _PREDEFINED_SPLITS_WFD[self.task_key][self.name]
        meta["trace_root"] = osp.join(self.data_root, trace_root) \
            if "://" not in trace_root else trace_root
        meta["json_file"] = osp.join(self.data_root, json_file) \
            if "://" not in trace_root else osp.join(trace_root, json_file)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_WFD["evaluator_type"][self.task_key]

        return meta

    def evaluate(self, predictions):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        return self.dataset_dicts




