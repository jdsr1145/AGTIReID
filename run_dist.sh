#!/bin/bash
root_dir=/data2/qianruiheng/MyFolder_qrh/LargeData
tau=0.015 
margin=0.1
noisy_rate=0.0  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=CUHK-PEDES
# CUHK-PEDES ICFG-PEDES RSTPReid

# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO 
# export PYTHONFAULTHANDLER=1 
# export CUDA_LAUNCH_BLOCKING=1 
# export DEBUG=1

N_GPU=4
GPUS_TO_USE=4,5,6,7

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=${GPUS_TO_USE} \
python -m  \
    torch.distributed.launch --nproc_per_node=$N_GPU --master_port=29501 train.py \
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
    --num_epoch 60
 