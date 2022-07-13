#!/bin/bash

FASTA=$1
OUT=$2

python predict_sav.py -i $FASTA -o $OUT
python render_mutation_movie.py $FASTA $OUT
