#!/bin/bash

set -e

i=0

rm -rf /tmp/logdir_opt
mkdir -p /tmp/logdir_opt
epochs=2
lr=1e-3

for optimizer in adam adamw adagrad adabelief adamax adamaxw amsgrad fromage lamb lars lion noisy_sgd novograd optimistic_gradient_descent dpsgd radam rmsprop sgd sm3 yogi; do
    ((i++))
    logdir=/tmp/logdir_opt/r"${optimizer}_${i}"
    mkdir -p ${logdir}

    echo ${i}, ${lr}, ${optimizer}
    python jax_model.py --config=jax_config.py  \
	   --config.model.num_blocks=1 \
	   --config.model.num_filters=64  \
	   --config.model.num_top=0 \
	   --config.train.optimizer=${optimizer} \
	   --config.epochs=${epochs}  \
	   --logdir=${logdir} \
	   --config.train.lr=${lr} > ${logdir}/out.txt 2> ${logdir}/err.txt
done
