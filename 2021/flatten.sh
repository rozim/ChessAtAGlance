
export LD_LIBRARY_PATH=/usr/local/lib:/opt/homebrew/opt/snappy/lib:/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow
for n in 0 1 2 3 4 5 6 7 8 9; do
    ./flatten stockfish-d1-${n}.leveldb stockfish-d1-${n}.snappy
    ./flatten stockfish-d3-${n}.leveldb stockfish-d3-${n}.snappy    
done
