import os
import torch

raw_path = 'data/pretrained/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

# download it from https://drive.google.com/file/d/1W6jdqziYqAqiyYide9SxvHE2y79A0KJC/view?usp=sharing
refinemask_path = 'data/pretrained/r101-coco-2x.pth'


merge_path = 'data/pretrained/refine_mask_htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

raw_ckpt = torch.load(raw_path, map_location='cpu')
refinemask_ckpt = torch.load(refinemask_path, map_location='cpu')
merge_ckpt = {}


for k, v in raw_ckpt['state_dict'].items():
    merge_ckpt[k] = v
for k, v in refinemask_ckpt['state_dict'].items():
    if 'roi_head.mask_head' in k:
        k_ = k.replace('roi_head.mask_head', 'roi_head.mask_head.0')
        merge_ckpt[k_] = v
        k_ = k.replace('roi_head.mask_head', 'roi_head.mask_head.1')
        merge_ckpt[k_] = v
        k_ = k.replace('roi_head.mask_head', 'roi_head.mask_head.2')
        merge_ckpt[k_] = v

torch.save(merge_ckpt, merge_path)