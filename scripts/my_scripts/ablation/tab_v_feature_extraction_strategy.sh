#!/bin/bash


# # experiments and datasets meta
EXPERIMENTS_HOME="./logs_mf"
KITTI_HOME="/your_dir/kitti2015/"
KITTI_RAW_HOME="/your_dir/kitti_raw/"

echo "KITTI_HOME: $KITTI_HOME"
echo "KITTI_RAW_HOME: $KITTI_RAW_HOME"
# model
MODEL=Mono_SF_MF #  Please comment out the position-enhanced motion feature aggregation in the model.
# Modify the feature extraction strategy in `BasicMotionEncoder` within `models/my_models/update.py`.



# save path
ALIAS=""
TIME=$(TZ=UTC-8 date +"%Y%m%d-%H-%M-%S")
SAVE_PATH="$EXPERIMENTS_HOME/ablation/tab_v_feature_extraction_strategy/$ALIAS$TIME"
CHECKPOINT='None'
# Loss and Augmentation
Train_Dataset=KITTI_Raw_4f
Train_Augmentation=Augmentation_SceneFlow_MF
Train_Loss_Function=Loss_SceneFlow_SelfSup_MF 

Valid_Dataset=KITTI_2015_Train_Full_mnsf_4f
Valid_Augmentation=Augmentation_Resize_Only_MV
Valid_Loss_Function=Eval_SceneFlow_KITTI_Train_MV # Eval_SceneFlow_KITTI_Train

EPOCHS=18
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2  --rdzv_endpoint=localhost:58149 ../../../main.py \
--batch_size=2 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=CosineAnnealingLR \
--lr_scheduler_T_max=$EPOCHS \
--model=$MODEL \
--num_workers=4 \
--optimizer=AdamW \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=1e-4 \
--optimizer_eps=1e-6 \
--save=$SAVE_PATH \
--total_epochs=$EPOCHS \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_augmentation_resize="[256, 832]" \
--training_dataset=$Train_Dataset \
--training_dataset_root=$KITTI_RAW_HOME \
--training_dataset_flip_augmentations=True \
--training_dataset_preprocessing_crop=True \
--training_dataset_num_examples=-1 \
--training_key=total_loss \
--training_loss=$Train_Loss_Function \
--validation_augmentation=$Valid_Augmentation \
--validation_augmentation_imgsize="[256, 832]" \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_HOME \
--validation_dataset_preprocessing_crop=False \
--validation_key=sf \
--validation_loss=$Valid_Loss_Function \
--tbIter 1500 \
--mixed_precision=True \
--iters 10 \
# --start_occ_reg=10