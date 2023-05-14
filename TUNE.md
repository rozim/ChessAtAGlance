
2023-05-14
==========

set executions=2 for (l2,max gradient)
very slow now, like 3h/run
the oracle thing didn't explore all the gradient values, so
may need to run again with max_gradients having more values between
0.1 and 1, and executions=0

this is what I had:
l2_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-9, 1e-10]
max_gradients = [10.0, 1.0, 0.1, 0.0]
..
epochs = 100
steps_per_epoch = 25
val_steps = 50
...
num_filters = 192
num_layers = 7
top_tower = [ 512 ]



2023-05-11
==========

layers/filters/flatten_1x1

best overall

	do_flatten_1x1		False  			2023-05-11
	top_tower		512
	num_layers		7
	num_filters		192

general pattern
	do_flatten_1x1 must be off - will be removed from model.py
	numn_layers=7 seemns like the sweet spot, though the bayesian alg may have focused on that
	filters=192 best, and higher better, though this was the max tested
	for top_tower, 512 and then 768 are best -- only tried 1 layer

num_filters_list = [32, 48, 64, 96, 112, 128, 144, 160, 192]
num_layers_list =  [4, 5, 6, 7, 8, 9, 10, 12]
top_tower_list = [ 128, 256, 512, 768, 1024, 1280, 1536 ]


2023-05-08
==========

Adam

amsgrad: False
beta_1: 0.60 (or 0.50)
beta_2: 0.995 (or 0.990)
epsilon: 1e-10


2023-05-07
==========

lr tuning
strong preference for
initial lr:       0.0050000
max decay factor: 0.10000

2023-05-06
==========

Activation results - see tune/*.csv

elu 5 of top 6.
silu high.
linear surprisingly good
selu not sampled enough
gelu worse than elu
softmax, sigmoid, hard_sigmoid, log_softmax are disasters
