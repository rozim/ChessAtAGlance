
nice time python select_random_positions_moves_simple.py --n=1000000 --pgn=../ChessData/mega-clean-2400.pgn  > data/mega-2400.csv
...
     4101.80 real      4087.83 user         3.08 sys


wc -l data/mega-2400.csv
  647209 data/mega-2400.csv


nice -19 time python generate_cnn_training_data_simple.py --csv=data/mega-2400.csv --shards=10 --out=data/mega-2400

150.8s 600000
n_game:  0
n_move:  0
n_dup:  0
n_gen:  647209
      166.07 real       166.41 user         3.10 sys

(venv-jax) (base) dave@macbook-pro ChessAtAGlance % ls -l -h data/*2400*
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00000-of-00010.recordio
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00001-of-00010.recordio
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00002-of-00010.recordio
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00003-of-00010.recordio
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00004-of-00010.recordio
-rw-r--r--  1 dave  staff   7.3M Dec 13 10:40 data/mega-2400-00005-of-00010.recordio
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00006-of-00010.recordio
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00007-of-00010.recordio
-rw-r--r--  1 dave  staff   7.2M Dec 13 10:40 data/mega-2400-00008-of-00010.recordio
-rw-r--r--  1 dave  staff   7.3M Dec 13 10:40 data/mega-2400-00009-of-00010.recordio
-rw-r--r--  1 dave  staff    39M Dec 13 10:15 data/mega-2400.csv

mkdir -p /tmp/logdir/2400
scripts/jax-2400.sh
