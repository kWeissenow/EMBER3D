#!/bin/bash

FASTA=$1
OUT=$2

python predict_sav.py -i $FASTA -o $OUT --save-distance-array --no-distance-map
python render_mutation_movie.py $FASTA $OUT
