# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .cbnet_cascade_rcnn import CBNetCascadeRCNN


@DETECTORS.register_module()
class CBNetHybridTaskCascade(CBNetCascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs):
        super(CBNetHybridTaskCascade, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
