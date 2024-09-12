#!/bin/bash

cd ..
# rm -r output/

# custom config
DATA=/DATASETS
TRAINER=Caption_distill_double

DATASET=coco2014_distill
CFG=rn50_coco2014  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
run_ID=finetune_coco
TRAIN_DATA=../data/finetune_coco_voc_20w.txt
lr=5e-5
lr_mul=0.1
noise=0.04
global_size=224
local_size=224
pre_epoch=40
w=50

export CUDA_VISIBLE_DEVICES=1

for SEED in 1
do
    DIR=output_tmp/${run_ID}/${global_size}x${local_size}xpretrain_${w}w_size224/
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job andsave the output to ${DIR}"
        python train_caption.py \
        --root ${DATA} \
        --seed ${SEED} \
        --global-size ${global_size} \
        --local-size ${local_size} \
        --pre-epoch ${pre_epoch} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --use-chatglm 1 \
        --train_data ${TRAIN_DATA} \
        --lr ${lr} \
        --lr_mul ${lr_mul} \
        --w ${w} \
        --noise ${noise} \
        --train_epoch 20 \
        --image_prompt_pretrain_load \
        TRAINER.Caption.N_CTX ${NCTX} \
        TRAINER.Caption.CSC ${CSC} \
        TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP}
    fi
done