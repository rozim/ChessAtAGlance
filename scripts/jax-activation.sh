#!/bin/bash

set -e

i=0


rm -rf /tmp/logdir_act
mkdir -p /tmp/logdir_act
epochs=50
lr=1e-3

for activation in relu elu silu celu selu gelu glu hard_sigmoid relu6 hard_silu softplus soft_sign sigmoid; do
    ((i++))
    logdir=/tmp/logdir_act/r"${activation}_${i}"
    mkdir -p ${logdir}
    start_time=$(date +"%s")

    python jax_model.py --config=jax_config.py  \
	   --config.model.num_blocks=1 \
	   --config.model.num_filters=64  \
	   --config.model.num_top=0 \
	   --config.model.activation=${activation} \
	   --config.epochs=${epochs}  \
	   --logdir=${logdir} \
	   --config.train.lr=${lr} > ${logdir}/out.txt 2> ${logdir}/err.txt

    end_time=$(date +"%s")
    dt=$((end_time - start_time))
    echo ${i}, ${dt}s, ${lr}, ${activation}
done
