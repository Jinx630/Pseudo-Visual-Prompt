#!/bin/bash

cd ..

# custom config
DATA=/DATASETS
TRAINER=Caption_distill_double

DATASET=coco2014_distill
CFG=rn50_coco2014  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
run_ID=pretrain_coco

global_size=224
local_size=224
pre_data=50

for SEED in 1
do
    DIR=output_tmp/${run_ID}/${global_size}x${local_size}*${pre_data}w/
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train_caption.py \
        --root ${DATA} \
        --train-data ../data/pretrain_${pre_data}w_coco_voc.txt \
        --global-size ${global_size} \
        --local-size ${local_size} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.Caption.N_CTX ${NCTX} \
        TRAINER.Caption.CSC ${CSC} \
        TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP}
    fi
done
