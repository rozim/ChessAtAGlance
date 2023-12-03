

i=0

rm -rf /tmp/logdir
mkdir -p /tmp/logdir

for lr in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 5e-1 5e-2 5e-3 5e-4 5e-5 5e-6; do
    ((i++))
    logdir=/tmp/logdir/r"${i}"
    mkdir -p ${logdir}

    echo $i, $lr
    python jax_model.py --config=jax_config.py  \
	   --config.model.num_blocks=1 \
	   --config.model.num_filters=64  \
	   --config.model.num_top=0 \
	   --config.epochs=25  \
	   --logdir=${logdir} \
	   --config.lr=${lr} > ${logdir}/out.txt 2> ${logdir}/err.txt
done
