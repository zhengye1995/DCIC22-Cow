#! /bin /bash

chmod +x tools/dist_test.sh

python tools/generate_testB.py

# inference bbox results for all models
./tools/dist_test.sh configs/cow/testB/htc_cbv2_swin_base.py /data/user_data/work_dirs/cb_swin_base.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testb/htc_cb_swin_base"

./tools/dist_test.sh configs/cow/testB/htc_cbv2_swin_large.py /data/user_data/work_dirs/cb_swin_large.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testb/htc_cb_swin_large"

./tools/dist_test.sh configs/cow/testB/cb_swin_small.py /data/user_data/work_dirs/cow_swin_small.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testb/cb_swin_small"

mkdir -p /data/user_data/results_testb/ensemble

cp /data/user_data/results_testb/*.json /data/user_data/results_testb/ensemble/

# ensemble to obtain proposals
python tools/post_process/wbf.py

# inference segm results with wbf proposals for all models
./tools/dist_test.sh configs/cow/testB/ensemble/htc_cbv2_swin_base.py /data/user_data/work_dirs/cb_swin_base.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testb/htc_cb_swin_base_prob"

./tools/dist_test.sh configs/cow/testB/ensemble/htc_cbv2_swin_large.py /data/user_data/work_dirs/cb_swin_large.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testb/htc_cb_swin_large_prob"

./tools/dist_test.sh configs/cow/testB/ensemble/cb_swin_small.py /data/user_data/work_dirs/cow_swin_small.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testb/cb_swin_small_prob"

mkdir -p /data/user_data/results_testb/ensemble_tmp

mv /data/user_data/results_testb/*.pkl /data/user_data/results_testb/ensemble_tmp

# ensemble to obtain final segms
python tools/post_process/ensemble_prob_masks.py

# convert format to submit
python tools/post_process/json2submit.py

mkdir -p /data/prediction_result & cp submit/result.json /data/prediction_result/