import os
import torch

raw_path = 'data/pretrained/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'
refinemask_path = 'data/pretrained/r101-coco-2x.pth'

merge_path = 'data/pretrained/refine_one_stage_mask_htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

raw_ckpt = torch.load(raw_path, map_location='cpu')
refinemask_ckpt = torch.load(refinemask_path, map_location='cpu')
merge_ckpt = {}

map_dict = {
    "roi_head.mask_head.2.stages.0.instance_logits.weight": "roi_head.mask_head.2.stage_instance_logits.0.weight",
    "roi_head.mask_head.2.stages.0.instance_logits.bias": "roi_head.mask_head.2.stage_instance_logits.0.bias",
    "roi_head.mask_head.2.stages.1.instance_logits.weight": "roi_head.mask_head.2.stage_instance_logits.1.weight",
    "roi_head.mask_head.2.stages.1.instance_logits.bias": "roi_head.mask_head.2.stage_instance_logits.1.bias",
    "roi_head.mask_head.2.stages.2.instance_logits.weight": "roi_head.mask_head.2.stage_instance_logits.2.weight",
    "roi_head.mask_head.2.stages.2.instance_logits.bias": "roi_head.mask_head.2.stage_instance_logits.2.bias",
    "roi_head.mask_head.2.final_instance_logits.weight": "roi_head.mask_head.2.stage_instance_logits.3.weight",
    "roi_head.mask_head.2.final_instance_logits.bias": "roi_head.mask_head.2.stage_instance_logits.3.bias",
}

for k, v in raw_ckpt['state_dict'].items():
    merge_ckpt[k] = v
for k, v in refinemask_ckpt['state_dict'].items():
    if 'roi_head.mask_head' in k:
        k_ = k.replace('roi_head.mask_head', 'roi_head.mask_head.2')
        # merge_ckpt[k_] = v
        if 'instance_logit' in k_:
            print(k_)
            k_new = map_dict[k_]
            merge_ckpt[k_new] = v
        else:
            merge_ckpt[k_] = v

torch.save(merge_ckpt, merge_path)