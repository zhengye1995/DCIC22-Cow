# 2022数字中国算法牛只分割图像竞赛方案代码



## 队伍名称：default(觅牛分队)

testA: 0.74643643

testB: 0.81680282

## 1. 解题方案整体思路：

+ Detector: 
    - Backbone: CBNetV2+Swin Transformer
    - RCNN Head:
        - box head:
            - 4Conv+1FC
            - GIou Loss
        - mask head:
            - HTC
            - mask loss x3
    - Post-process: soft-nms
+ Data augmentation:
    - custom CopyPaste
    - multi-scale training and testing
    - random flip (direction=['horizontal', 'vertical', 'diagonal'])
    - CutOut
    - PhotoMetricDistortion
    - GridMask
+ Model ensemble:
    - coarse-to-fine
        - ensemble boxes
        - ensemble segms

## 2. 运行环境
+ 系统 ubuntu 18.01 
+ python3.7 或 3.8，其他的python依赖包均已安装在docker中
+ torch 1.8 或者 1.7.1
+ cuda 11.1
+ cudnn 8
+ GPU： 3090 x 4
+ 硬盘：100GB 容量以上
+ 特殊环境：
    - mmcv-full==1.4.0

## 3. 文件目录说明
进行训练测试时需要的具体的文件如下：
训练好的模型文件可以从[百度网盘](https://pan.baidu.com/s/1PaJB02zoHfP2SCuZ08XrpA?pwd=1ygr)下载
```
|-- data
	|-- user_data
	    |-- work_dirs/ 包含3个训练好的模型的模型文件
	    |-- annotations 包含推理或者训练过程中转换的coco格式文件，其中文件代码自动生成
        |-- pretrained 包含模型训练需要的公开的coco预训练模型，由于模型体积大，这里只给出了开源github的下载链接，可能下载下来的文件为zip压缩格式，需要解压后放在此目录下
        |-- 剩余文件夹均为训练或者预测过程中的中间文件
    |-- prediction_result
	    |-- result.json   按照比赛提交格式生成的提交文件
    |-- code
	    |-- 本repo所有代码放在该目录下
        |-- 其中 run.sh 为一键推理命令， train.sh 为一键训练命令
    |-- raw_data
        |-- 比赛的数据集文件（这里由官方提供，代码读取时按照和官方完全一致的目录来取用,即官网数据下载解压后的格式,所以需要将官方数据放置在这里）：
        |-- 训练数据目录：/data/raw_data/train_dataset/
            |-- 200/images
            |-- 200/data.json
        |-- 测试数据目录：/data/raw_data/test_dataset_A/ 以及 /data/raw_data/test_dataset_B/
```
## 4. 推理得到B榜提交结果的运行说明
+ 按照上述过程下载3个模型文件并放置于/data/user_data/work_dirs
+ run.sh 为一键推理命令

## 5. 模型训练的运行说明
+ 首先下载三个开源预训练模型，这里给出对应的下载链接：
    - https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth.zip
    - https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip
    - https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip
+ 下载上述3个预训练模型后，进行解压，最终 data/user_data/pretrained 目录下应该有三个.pth后缀的预训练模型文件：
    - htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth
    - htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth
    - cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth
+  train.sh 为一键训练命令

## 6. Author

rill：18813124313@163.com
