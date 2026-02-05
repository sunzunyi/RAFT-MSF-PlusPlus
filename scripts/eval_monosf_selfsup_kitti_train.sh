#!/bin/bash

# DATASETS_HOME
KITTI_HOME="/your_dir/kitti2015/"
KITTI_RAW_HOME="/your_dir/kitti_raw/"
CHECKPOINT="ckpt/raft_msf-pp.ckpt"

# model
MODEL=Mono_SF_MF 

Valid_Dataset=KITTI_2015_Train_Full_mnsf_4f # KITTI_2015_Train_Full_mnsf_4f
Valid_Augmentation=Augmentation_Resize_Only_MV # Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Train_MV  #  Eval_SceneFlow_KITTI_Test   Eval_SceneFlow_KITTI_Train_MV

# training configuration
# save path
ALIAS="-raft-msf-offical-kitti-train-"
TIME=$(TZ=UTC-8 date +"%Y%m%d-%H-%M-%S")
SAVE_PATH="eval/$MODEL/$ALIAS$TIME"
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--evaluation=True \
--num_workers=4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--iters 10 \
--save_flow=False \
--save_disp=False \
--save_disp2=False