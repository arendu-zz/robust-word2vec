#!/bin/sh
set -e
CORPUS=$1
OUT_CORPUS=$2
WINDOW=$3
FREQ=$4
./extract_wc_pairs.py -c $CORPUS -o $OUT_CORPUS -w $WINDOW -f $FREQ -i
for DATA_TYPE in 'cbow' 'sg'
do
    OUT_FILE=$OUT_CORPUS.$DATA_TYPE.txt
    gsplit -n 4  -a 1 -d $OUT_FILE $OUT_FILE.part
    cat $OUT_FILE.part0 > $OUT_FILE.train
    cat $OUT_FILE.part1 >> $OUT_FILE.train
    trash $OUT_FILE.part0
    trash $OUT_FILE.part1
    mv $OUT_FILE.part2  $OUT_FILE.dev
    mv $OUT_FILE.part3  $OUT_FILE.test
done



