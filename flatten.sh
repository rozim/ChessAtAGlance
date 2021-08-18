
export LD_LIBRARY_PATH=/usr/local/lib:/opt/homebrew/opt/snappy/lib:/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow
for n in 0 1 2 3 4 5 6 7 8 9; do
  ./flatten mega-v2-${n}.leveldb mega-v2-${n}.snappy
done
