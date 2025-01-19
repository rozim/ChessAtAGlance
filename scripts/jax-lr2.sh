#!/bin/bash

set -e

i=0

basedir=/tmp/logdir_lr4
rm -rf ${basedir}
mkdir -p ${basedir}
epochs=200
optimizer=lion
activation=glu
batch=1024

for lr in 5e-4 6e-4 7e-4 8e-4 9e-4 1e-4 3e-4 2e-4 4e-4 1e-3 1e-5 5e-5; do
    ((i++))
    logdir=${basedir}/"${lr}"
    mkdir -p ${logdir}
    start_time=$(date +"%s")

    python jax_model.py --config=jax_config.py  \
	   --config.model.num_blocks=1 \
	   --config.model.num_filters=64  \
	   --config.model.num_top=0 \
	   --config.model.activation=${activation} \
	   --config.train.optimizer=${optimizer} \
	   --config.epochs=${epochs}  \
	   --config.batch_size=${batch} \
	   --logdir=${logdir} \
	   --config.train.lr=${lr} > ${logdir}/out.txt 2> ${logdir}/err.txt

    end_time=$(date +"%s")
    dt=$((end_time - start_time))
    echo ${i}, ${dt}s, ${lr}, ${activation}, ${optimizer}, ${batch}
done
