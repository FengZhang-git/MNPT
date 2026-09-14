#!/bin/bash

TRAINER=MNPT

CSC=False
CTP=end

DATA=$1
DATASET=$2
CFG=$3
LOAD_EPOCH=${CFG:10:2}
MODEL_dir=$4
Output_dir=$4
POSITIVE_WEIGHTS=$5
NCTXNP=$6
min_temperature=1
T=1
NEGT=1
NCTXNP=16
NCTXNEG=16
alpha=0.015

python eval_ood_detection.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/MNPTPLUS/${CFG}.yaml \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--load-epoch ${LOAD_EPOCH} \
--T ${T} \
--NEGT ${NEGT} \
--min_temperature ${min_temperature} \
--alpha_value ${alpha} \
TRAINER.MNPT.N_CTX_NEG ${NCTXNEG} \
TRAINER.MNPT.N_CTX_NP ${NCTXNP} \
TRAINER.MNPT.POSITIVE_WEIGHTS ${POSITIVE_WEIGHTS}
