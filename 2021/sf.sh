
for n in 0 1 2 3 4 5 6 7 8 9; do
    echo $n
    python leveldb_to_recordio.py --fn_in=stockfish-v5-d1-${n}.leveldb --fn_out=stockfish-v5-d1-${n}.recordio
    python leveldb_to_recordio.py --fn_in=stockfish-v5-d3-${n}.leveldb --fn_out=stockfish-v5-d3-${n}.recordio
done
