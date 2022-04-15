import json
import cv2
import numpy as np
import os
import pycocotools.mask as mask
from tqdm import tqdm

test_json = 'data/annotations/testA.json'
# bbox_results = 'results/cas_dcn_mask_r50.bbox.json'
segm_results = 'results/cascade_mask_convnext_xlarge_softnms_tta_001_36e.segm.json'

def polygonFromMask(maskedArr):
      # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
      contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      segmentation = []
      valid_poly = 0
      for contour in contours:
      # Valid polygons have >= 6 coordinates (3 points)
         if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
      if valid_poly == 0:
         raise ValueError
      return segmentation

def mask2polygon(mask):
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:# and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    shape=polygons.shape
    polygons=polygons.reshape(shape[0],-1,2)
    cv2.fillPoly(mask, polygons,color=1) # 非int32 会报错
    return mask


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]
    return mask.reshape(height, width).T

with open(test_json, 'r') as f:
    test_info = json.load(f)

# with open(bbox_results, 'r') as f:
#     bbox_anno = json.load(f)

with open(segm_results, 'r') as f:
    segm_anno = json.load(f)

imageid2infos = {}
for image in test_info['images']:
    imageid2infos[image['id']] = os.path.join("images", os.path.basename(image['file_name']))

final_res = []
anno_id = 1
for anno in tqdm(segm_anno):
    image_id = anno['image_id']
    imageinfo = imageid2infos[image_id]
    # anno_id = anno['id']
    final_res.append({
        "image_id": imageinfo,
        "bbox": anno['bbox'],
        "segmentation": anno['segmentation'],
        "category_id": 1,
        "score": anno['score']
    })
with open('submit/cascade_mask_convnext_xlarge_softnms_tta_001_36e.json', 'w') as f:
    json.dump(final_res, f)
