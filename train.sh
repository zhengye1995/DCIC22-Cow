
chmod +x tools/dist_train.sh

# generate seg map
python tools/post_process/generate_seg_map.py

./tools/dist_train.sh configs/cow/htc_cbv2_swin_base.py 4 --no-validate

./tools/dist_train.sh configs/cow/htc_cbv2_swin_large.py 4 --no-validate

./tools/dist_train.sh configs/cow/cb_swin_small.py 2 --no-validate
