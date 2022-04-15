import os
import torch

raw_path = 'data/pretrained/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'
ms_score_path = 'data/pretrained/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth'

merge_path = 'data/pretrained/ms_scoring_htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

raw_ckpt = torch.load(raw_path, map_location='cpu')
refinemask_ckpt = torch.load(ms_score_path, map_location='cpu')
merge_ckpt = {}


for k, v in raw_ckpt['state_dict'].items():
    merge_ckpt[k] = v
for k, v in refinemask_ckpt['state_dict'].items():
    if 'roi_head.mask_iou_head' in k:
        merge_ckpt[k] = v

torch.save(merge_ckpt, merge_path)