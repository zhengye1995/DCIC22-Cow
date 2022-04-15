import json
import cv2
import numpy as np
import os
import pycocotools.mask as mask
from tqdm import tqdm

# test_json = 'data/annotations/testA.json'
test_json = '/data/user_data/annotations/testB.json'
segm_results = '/data/user_data/results_testb/testb.segm.json'


with open(test_json, 'r') as f:
    test_info = json.load(f)

with open(segm_results, 'r') as f:
    segm_anno = json.load(f)

imageid2infos = {}
for image in test_info['images']:
    imageid2infos[image['id']] = os.path.join("images", os.path.basename(image['file_name']))

final_res = []
anno_id = 1
print("raw:", len(segm_anno))
for anno in tqdm(segm_anno):
    image_id = anno['image_id']
    imageinfo = imageid2infos[image_id]
    _, _, w, h = anno['bbox']
    if w < 30 or h < 30:
        continue
    final_res.append({
        "image_id": imageinfo,
        "bbox": anno['bbox'],
        "segmentation": anno['segmentation'],
        "category_id": 1,
        "score": anno['score']
    })
print('filter:', len(final_res))
print(len(final_res)-len(segm_anno))
os.makedirs('submit', exist_ok=True)
with open('submit/result.json', 'w') as f:
    json.dump(final_res, f)