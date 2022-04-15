import os
import torch

raw_path = 'data/pretrained/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth'
htc_path = 'data/pretrained/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

merge_path = 'data/pretrained/htc_cbv2_swin_small_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_3x_coco.pth'

raw_ckpt = torch.load(raw_path, map_location='cpu')
refinemask_ckpt = torch.load(htc_path, map_location='cpu')
merge_ckpt = {}


for k, v in raw_ckpt['state_dict'].items():
    merge_ckpt[k] = v
for k, v in refinemask_ckpt['state_dict'].items():
    # if 'roi_head.mask_head' in k:  # HTC MASK head
    #     merge_ckpt[k] = v
    if 'roi_head.semantic_head' in k:  # HTC seg head
        merge_ckpt[k] = v

torch.save(merge_ckpt, merge_path)