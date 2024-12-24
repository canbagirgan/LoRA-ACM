#!/bin/bash

python3 eval.py \
    --label_path dataset/test.csv \
    --result_path model_responses/model_response.json \
    --outfile results/results.txt \
    --nrows \
    --msize 30