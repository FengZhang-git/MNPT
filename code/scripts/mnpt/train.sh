#!/bin/bash

TRAINER=MNPT

DATA=$1
DATASET=$2
CFG=$3
SHOTS=$4
NCTXNP=$5
miu=$6
POSITIVE_WEIGHTS=$7
annealed_rate=$8
NCTXNEG=16
min_temperature=1
init_temperature=10

for SEED in 1; do
    DIR=output/${DATASET}_${TRAINER}/${CFG}_${SHOTS}shots_anneal${annealed_rate}_${init_temperature}_np${NCTXNP}_nctnegx${NCTXNEG}_miu${miu}_t${min_temperature}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo $PWD
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/MNPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --miu_value ${miu} \
        --min_temperature ${min_temperature} \
        --topk ${topk} \
        --annealed_temperature \
        --annealed_rate ${annealed_rate} \
        --init_temperature ${init_temperature} \
        TRAINER.MNPT.N_CTX_NEG ${NCTXNEG} \
        TRAINER.MNPT.N_CTX_NP ${NCTXNP} \
        TRAINER.MNPT.POSITIVE_WEIGHTS ${POSITIVE_WEIGHTS} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
