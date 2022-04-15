# Copyright (c) OpenMMLab. All rights reserved.
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class FCNEdgeHead(BaseModule):

    def __init__(self,
                 loss_edge=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(FCNEdgeHead, self).__init__(init_cfg)

        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_edge)
        self.conv_soble = nn.Conv2d(1, 1, 3, bias=False)
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        self.conv_soble.weight.data = torch.from_numpy(sobel_kernel)
        self.conv_soble.weight.requires_grad = False

    @auto_fp16()
    def forward(self, x):
        edge_pred = self.conv_soble(x)
        return edge_pred

    def get_targets(self, gt_masks):
        edge_targets = gt_masks.masks
        edge_targets = self.conv_soble(edge_targets)
        return edge_targets

    @force_fp32(apply_to=('edge_pred', ))
    def loss(self, edge_pred, edge_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if edge_pred.size(0) == 0:
            loss_mask = edge_pred.sum()
        else:
            loss_mask = self.loss_edge(edge_pred, edge_targets,
                                       torch.zeros_like(labels))
        loss['loss_mask'] = loss_mask
        return loss
