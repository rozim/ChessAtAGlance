

for n in 0 1 2 3 4 5 6 7 8 9; do
    python leveldb_to_recordio.py --fn_in=mega-v5-${n}.leveldb --fn_out=mega-v5-${n}.recordio
done
