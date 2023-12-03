

i=0

rm -rf /tmp/logdir
mkdir -p /tmp/logdir
epochs=4
lr=1e-3

for blocks in 0 1 2 3 4; do
    for filters in 1 16 32 64 128; do
	((i++))
	logdir=/tmp/logdir/r"${i}"
	mkdir -p ${logdir}

	echo $i, $lr, b=${blocks} f=${filters}
	python jax_model.py --config=jax_config.py  \
	   --config.model.num_blocks=${blocks} \
	   --config.model.num_filters=${filters}  \
	   --config.model.num_top=0 \
	   --config.epochs=${epochs}  \
	   --logdir=${logdir} \
	   --config.lr=${lr} > ${logdir}/out.txt 2> ${logdir}/err.txt
    done
done
