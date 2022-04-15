# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .refine_mask_head import SimpleRefineMaskHead
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32


@HEADS.register_module()
class RefineHTCMaskHead(SimpleRefineMaskHead):

    def __init__(self, conv_out_channels, with_conv_res=True, *args, **kwargs):
        super(RefineHTCMaskHead, self).__init__(*args, **kwargs)
        self.conv_out_channels = conv_out_channels
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    @auto_fp16()
    def forward(self, x, res_feat=None, semantic_feat=None, rois=None,
                roi_labels=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            x = x + res_feat

        for conv in self.instance_convs:
            instance_feats = conv(x)
        res_feat = instance_feats
        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        stage_instance_preds = []
        for idx, stage in enumerate(self.stages):
            instance_logits = self.stage_instance_logits[idx](instance_feats)[torch.arange(len(rois)), roi_labels][:,
                              None]
            upsample_flag = self.pre_upsample_last_stage or idx < len(self.stages) - 1
            instance_feats = stage(instance_feats, instance_logits, semantic_feat, rois, upsample_flag)
            stage_instance_preds.append(instance_logits)

        # if use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        instance_preds = self.stage_instance_logits[-1](instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        if not self.pre_upsample_last_stage:
            instance_preds = F.interpolate(instance_preds, scale_factor=2, mode='bilinear', align_corners=True)
        stage_instance_preds.append(instance_preds)

        outs = []
        if return_logits:
            outs.append(stage_instance_preds)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
