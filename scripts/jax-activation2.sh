#!/bin/bash

set -e

i=0

basedir=/tmp/logdir_act2
rm -rf ${basedir}
mkdir -p ${basedir}
epochs=100
lr=1e-3
optimizer=lion

# hard_sigmoid
# sigmoid

for activation in gelu hard_silu relu elu silu celu selu  glu relu6 softplus soft_sign ; do
    ((i++))
    logdir=${basedir}/"${activation}_${i}"
    mkdir -p ${logdir}
    start_time=$(date +"%s")

    python jax_model.py --config=jax_config.py  \
	   --config.model.num_blocks=1 \
	   --config.model.num_filters=64  \
	   --config.model.num_top=0 \
	   --config.model.activation=${activation} \
	   --config.train.optimizer=${optimizer} \
	   --config.epochs=${epochs}  \
	   --logdir=${logdir} \
	   --config.train.lr=${lr} > ${logdir}/out.txt 2> ${logdir}/err.txt

    end_time=$(date +"%s")
    dt=$((end_time - start_time))
    echo ${i}, ${dt}s, ${lr}, ${activation}
done
