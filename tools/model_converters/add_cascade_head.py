import os
import torch

htc_path = 'data/pretrained/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

save_path = 'data/pretrained/htc_cbv2_swin_base22k_patch4_window7_4head_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

raw_ckpt = torch.load(htc_path, map_location='cpu')
save_ckpt = {}
for k, v in raw_ckpt['state_dict'].items():
    save_ckpt[k] = v
for k, v in raw_ckpt['state_dict'].items():
    if 'roi_head.bbox_head.2' in k:
        print(k)
        k_ = k.replace('roi_head.bbox_head.2', 'roi_head.bbox_head.3')
        save_ckpt[k_] = v
    if 'roi_head.mask_head.2' in k:  # HTC MASK head
        print(k)
        k_ = k.replace('roi_head.mask_head.2', 'roi_head.mask_head.3')
        save_ckpt[k_] = v

torch.save(save_ckpt, save_path)