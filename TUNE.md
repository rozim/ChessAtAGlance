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
