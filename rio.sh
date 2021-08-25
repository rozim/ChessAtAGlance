

for n in 0 1 2 3 4 5 6 7 8 9; do
    python snappy_to_recordio.py --fn_in=mega-v3-${n}.snappy      --fn_out=mega-v3-${n}.recordio
    python snappy_to_recordio.py --fn_in=stockfish-d1-${n}.snappy --fn_out=stockfish-d1-${n}.recordio
    python snappy_to_recordio.py --fn_in=stockfish-d3-${n}.snappy --fn_out=stockfish-d3-${n}.recordio    
done

