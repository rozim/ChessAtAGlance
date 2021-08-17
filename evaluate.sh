

for n in 0 1 2 3 4 5 6 7 8 9; do
    python evaluate.py --plan=v0.toml --model=results/2021-08-17_10:30:34/my.model --fn=mega-v2-${n}.leveldb --n=1024
    mv -i evaluate.csv evaluate-${n}.csv
done    
