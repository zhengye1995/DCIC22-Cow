import os
import torch

# raw_path = 'data/pretrained/cascade_mask_rcnn_convnext_base_22k_3x.pth'
raw_path = 'data/pretrained/cascade_mask_rcnn_convnext_large_22k_3x.pth'
# raw_path = 'data/pretrained/cascade_mask_rcnn_convnext_xlarge_22k_3x.pth'
htc_path = 'data/pretrained/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'

merge_path = 'data/pretrained/htc_cascade_mask_rcnn_convnext_large_22k_3x.pth'

raw_ckpt = torch.load(raw_path, map_location='cpu')
htc_ckpt = torch.load(htc_path, map_location='cpu')
merge_ckpt = {}


for k, v in raw_ckpt['state_dict'].items():
    merge_ckpt[k] = v
for k, v in htc_ckpt['state_dict'].items():
    # if 'roi_head.mask_head' in k:  # HTC MASK head
    #     merge_ckpt[k] = v
    if 'roi_head.semantic_head' in k:  # HTC seg head
        merge_ckpt[k] = v
print(merge_ckpt.keys())
if "roi_head.semantic_head.lateral_convs.0.conv.weight" in merge_ckpt:
    print(True)
torch.save(merge_ckpt, merge_path)