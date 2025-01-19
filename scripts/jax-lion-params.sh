#!/bin/bash

set -e

i=0

basedir=/tmp/logdir_lion_v4
rm -rf ${basedir}
mkdir -p ${basedir}
epochs=250
optimizer=lion
activation=glu
batch=1024
lr=6e-4

for b1 in 0.80 85 0.89 0.90 0.91; do
    for b2 in 0.990 0.980 0.985 0.975; do
	for weight_decay in 1e-3 2e-3 3e-3 1.5e-3; do
	    ((i++))
	    logdir=${basedir}/"${b1}_${b2}_${weight_decay}"
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
		   --config.optimizer.lion.b1=${b1} \
		   --config.optimizer.lion.b2=${b2} \
		   --config.optimizer.lion.weight_decay=${weight_decay} \
		   --config.train.lr=${lr} \
		   --logdir=${logdir} \
			    > ${logdir}/out.txt 2> ${logdir}/err.txt

	    end_time=$(date +"%s")
	    dt=$((end_time - start_time))
	    echo ${i}, ${dt}s, ${lr}, ${activation}, ${optimizer}, ${batch}, ${b1}, ${b2}, ${weight_decay}
	done
    done
done
