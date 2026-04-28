#!/bin/bash
# Test homography STN
root_dir=/data2/qianruiheng/MyFolder_qrh/LargeData
tau=0.015 
margin=0.1
noisy_rate=0.0  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=TAGPEDES
# CUHK-PEDES ICFG-PEDES RSTPReid

# export DEBUG=1

# TODO: 补做Full RDE 实验

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=1 \
    python mytest.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name RDE \
    --img_aug \
    --txt_aug \
    --batch_size 64 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}  \
    --num_epoch 30 \
    --enable_tse \
    --enable_ccd \
    --pretrain_choice /data2/qianruiheng/MyFolder_qrh/AGReID/AGTIReID/data/best0.pth \
    # --scale_reexpress 0.2 \
    # --enable_reexpress \
    # --dualstream \
    # --source_aerial train_paired_aerial_only_annotated_ver2.json \
    # --enable_text_aerial \
    # --enable_vdt \
    # --scale_vdt 0.001 \
    
    
    # --enable_loss_view \
    # --scale_loss_view 0.2 \
    # --enable_reranking \
    # --enable_calssifier \
    # --scale_calssifier 0.1 \
    