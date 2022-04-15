import json
import os
from glob import glob
import pycocotools.mask as maskutil
import numpy as np
import mmcv

def rle2mask(rle):
    return maskutil.decode(rle)

def mask2rle(mask):
    return maskutil.encode(mask)


def ensemble_masks(submit_path, save_path, score_thr):
    # submit_paths = glob(submit_path + '/*.pkl')
    submit_paths = ['/data/user_data/results_testb/ensemble_tmp/htc_cb_swin_large_prob.segm_prob.pkl',
                    '/data/user_data/results_testb/ensemble_tmp/htc_cb_swin_base_prob.segm_prob.pkl',
                    '/data/user_data/results_testb/ensemble_tmp/cb_swin_small_prob.segm_prob.pkl'
                    ]
    print(submit_paths)
    results_all = []
    for pkl_file in submit_paths:
        results = mmcv.load(pkl_file)
        results_all.append(results)

    raw_json_path = '/data/user_data/results_testb/htc_cb_swin_large.segm.json' # raw segm results
    with open(raw_json_path, 'r') as f:
        raw_results = json.load(f)
    low_score_results = []
    for res in raw_results:
        if res['score'] < score_thr:
            low_score_results.append(res)

    json_path = '/data/user_data/results_testb/htc_cb_swin_large_prob.segm.json'  # generate with pkl
    with open(json_path, 'r') as f:
        final_results = json.load(f)
    instance_idx = 0

    for i in range(len(results_all[0])):  # 100 testa
        for j in range(len(results_all[0][i])):  # num_bbox <-> num_mask
            mask_prob = [results_all[0][i][j]]
            for k in range(1, len(results_all)):
                mask_ = results_all[k][i][j]   # k model, i image, j instance(bbox)
                mask_prob.append(mask_)
            mask_prob_final = np.mean(mask_prob, axis=0)
            mask_prob_final = mask_prob_final > 0.5
            rle_final = mask2rle(np.asfortranarray(mask_prob_final))
            rle_final['counts'] = rle_final['counts'].decode()
            temp = final_results[instance_idx]
            temp['segmentation'] = rle_final
            final_results[instance_idx] = temp
            instance_idx += 1

    final_results.extend(low_score_results)
    with open(save_path, 'w') as f:
        json.dump(final_results, f)


if __name__ == "__main__":
    submit_path = '/data/user_data/results_testb/ensemble_tmp'
    save_path = '/data/user_data/results_testb/testb.segm.json'
    ensemble_masks(submit_path, save_path, score_thr=0.05)
