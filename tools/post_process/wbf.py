# -*- coding: utf-8 -*-

import numpy as np
import os
import json
from glob import glob
from mmcv.ops import nms
from tqdm import tqdm
import torch
from mmdet.core import multiclass_nms
from ensemble_boxes import weighted_boxes_fusion

def get_all_img_in_json(json_path):

    with open(json_path, 'r') as load_f:
        json_data = json.load(load_f)

    all_images = []
    for i, box in enumerate(json_data):
        name = box['name']
        if name not in all_images:
            all_images.append(name)
    return all_images

def ensemble(submit_path, save_path, max_class_id, th_nmsiou, th_score, weights=None):
    submit_paths = glob(submit_path+'/*.json')
    # img_list = get_all_img_in_json(submit_paths[0])
    id2annos = {}
    final_result = []
    for json_file in submit_paths:
        with open(json_file, 'r') as f:
            result = json.load(f)
        for box in result:
            img_id = box['image_id']
            if img_id not in id2annos:
                id2annos[img_id] = []
            if weights is not None:
                box['score'] *= weights[os.path.basename(json_file)]
            id2annos[img_id].append(box)
            # category = box['category_id']
            # bbox = box['bbox']
            # score = box['score']
            # boxes.append(bbox + [score] + [category_id])
    nms_cfg = dict(type='nms', iou_thr=th_nmsiou)  # nms_cfg = dict(type='soft_nms', iou_thr=th_nms, min_score=0.01)
    # nms_cfg = dict(type='soft_nms', iou_thr=th_nmsiou, min_score=th_score)  # nms_cfg = dict(type='soft_nms', iou_thr=th_nms, min_score=0.01)
    for id, annos in id2annos.items():
        multi_scores = []
        boxes = []
        for anno in annos:
            box = anno['bbox']
            xmin = box[0]
            ymin = box[1]
            weight = box[2]
            height = box[3]
            confidence = anno["score"]
            label_class = anno["category_id"]
            scores = []
            for _ in range(max_class_id+1):
                scores.append(0)
            scores[int(label_class)] = confidence
            multi_scores.append(scores)
            boxes.append([xmin, ymin, xmin + weight, ymin + height])

        boxes = torch.from_numpy(np.array(boxes, dtype='float32'))
        multi_scores = torch.from_numpy(np.array(multi_scores, dtype='float32'))
        if boxes.shape[0] == 0:
            continue
        det_bboxes, det_labels = multiclass_nms(boxes, multi_scores, th_score, nms_cfg, 200)
        det_bboxes[:, 2] = det_bboxes[:, 2] - det_bboxes[:, 0]
        det_bboxes[:, 3] = det_bboxes[:, 3] - det_bboxes[:, 1]
        for i in range(det_bboxes.shape[0]):
            x, y, w, h, score = det_bboxes[i]
            x = round(float(x), 4)
            y = round(float(y), 4)
            w = round(float(w), 4)
            h = round(float(h), 4)
            score = float(score)
            label = int(det_labels[i])
            final_result.append({'image_id': id, "bbox":[x, y, w, h], "score":score, "category_id":label+1})
    with open(save_path, 'w') as fp:
        json.dump(final_result, fp, indent=1)


def ensemble_wbf(submit_path, save_path, th_nmsiou, th_score, conf_type, weights=None, imageid2size=None):
    # submit_paths = glob(submit_path + '/*.bbox.json')
    submit_paths = ['/data/user_data/results_testb/ensemble/htc_cb_swin_large.bbox.json',
                    '/data/user_data/results_testb/ensemble/htc_cb_swin_base.bbox.json',
                    '/data/user_data/results_testb/ensemble/cb_swin_small.bbox.json']
    print(submit_paths)
    model_nums = len(submit_paths)
    final_result = []
    id2annos = [{} for _ in range(model_nums)]
    model_weights = [1 for _ in range(model_nums)]
    for i, json_file in enumerate(submit_paths):
        if weights is not None:
            model_weights[i] = weights[os.path.basename(json_file)]
        with open(json_file, 'r') as f:
            result = json.load(f)
        for box in result:
            img_id = box['image_id']

            if img_id not in id2annos[i]:
                id2annos[i][img_id] = []
            id2annos[i][img_id].append(box)
    # weights = [1 for _ in range(model_nums)]
    iou_thr = th_nmsiou
    skip_box_thr = th_score
    for id, _ in id2annos[0].items():
        scores_list = [[] for _ in range(model_nums)]
        boxes_list = [[] for _ in range(model_nums)]
        labels_list = [[] for _ in range(model_nums)]
        img_id = id
        img_size = imageid2size[img_id]
        img_w, img_h = img_size

        for j in range(model_nums):
            if id in id2annos[j]:
                for anno in id2annos[j][id]:
                    box = anno['bbox']
                    xmin = box[0]
                    ymin = box[1]
                    width = box[2]
                    height = box[3]
                    xmax = xmin + width
                    ymax = ymin + height

                    xmax = xmax / img_w
                    xmin = xmin / img_w
                    ymin = ymin / img_h
                    ymax = ymax / img_h
                    confidence = anno["score"]
                    label_class = anno["category_id"]
                    scores_list[j].append(confidence)
                    boxes_list[j].append([xmin, ymin, xmax, ymax])
                    labels_list[j].append(label_class)

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=model_weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            score = float(scores[i])
            label = int(labels[i])
            x1 = round(float(x1*img_w), 4)
            y1 = round(float(y1*img_h), 4)
            x2 = round(float(x2*img_w), 4)
            y2 = round(float(y2*img_h), 4)
            final_result.append({'image_id': id, "bbox": [x1, y1, x2-x1, y2-y1], "score": score, "category_id": label})
    with open(save_path, 'w') as fp:
        json.dump(final_result, fp)

    return final_result


if __name__ == '__main__':
    submit_path = '/data/user_data/results_testb/ensemble'
    save_path = '/data/user_data/results_testb/htc_cb_swin_large_htc_cb_swin_base_cb_swin_small_wbf.json'

    # raw_testa_json = '/data/user_data/annotations/testA.json'
    raw_testa_json = '/data/user_data/annotations/testB.json'
    with open(raw_testa_json, 'r') as f:
        infos = json.load(f)
    imageid2size = {}
    for info in infos['images']:
        imageid2size[info['id']] = [info['width'], info['height']]

    weights = None

    # ensemble(submit_path, save_path, 5, th_nmsiou=0.7, th_score=0.005, weights=weights)
    final_result = ensemble_wbf(submit_path, save_path, th_nmsiou=0.7, th_score=0.00001, conf_type='max', weights=weights, imageid2size=imageid2size)
    filter_results = []
    for res in final_result:
        if res['score'] >= 0.05:
            filter_results.append(res)
    # save_path = 'results/htc_cb_swin_large_htc_cb_swin_base_cb_swin_small_wbf_filter.json'
    save_path = '/data/user_data/results_testb/htc_cb_swin_large_htc_cb_swin_base_cb_swin_small_wbf_filter.json'
    with open(save_path, 'w') as fp:
        json.dump(filter_results, fp)



