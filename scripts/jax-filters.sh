#!/bin/bash

set -e

i=0

basedir=/tmp/logdir_filters_b2_v1
rm -rf ${basedir}
mkdir -p ${basedir}
epochs=250
optimizer=lion
activation=glu
batch=1024
lr=6e-4
blocks=2

for filters in 2 4 8 16 32 64 128 256 ; do
    ((i++))
    logdir=${basedir}/"${blocks}_${filters}"
    mkdir -p ${logdir}
    start_time=$(date +"%s")

    python jax_model.py --config=jax_config.py  \
	   --config.model.num_blocks="${blocks}" \
	   --config.model.num_filters="${filters}"  \
	   --config.model.num_top=0 \
	   --config.model.activation=${activation} \
	   --config.train.optimizer=${optimizer} \
	   --config.epochs=${epochs}  \
	   --config.batch_size=${batch} \
	   --config.train.lr=${lr} \
	   --logdir=${logdir} \
	       > ${logdir}/out.txt 2> ${logdir}/err.txt

    end_time=$(date +"%s")
    dt=$((end_time - start_time))
    echo ${i}, ${dt}s, ${lr}, ${activation}, ${optimizer}, ${batch}, ${filters}, ${blocks}
done
