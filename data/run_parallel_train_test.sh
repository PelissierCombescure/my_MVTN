#!/bin/bash

# ./run_parallel_train_test.sh 12 circular /media/disk1/mpelissi-data/MVTN/circular-12/Projections

# Read input arguments
NB_VIEWS=$1
VIEW_CONFIG=$2
DIR_OUTPUT=$3

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <nb_views> <view_config> <dir_output>"
    exit 1
fi

mkdir -p logs

# Run train split
CUDA_VISIBLE_DEVICES=1 python3 run_projections_mvtn_parallel_test_train.py \
    --nb_views $NB_VIEWS \
    --view_config $VIEW_CONFIG \
    --dir_output $DIR_OUTPUT \
    --split train \
    > logs/train_${VIEW_CONFIG}_${NB_VIEWS}-aligned.log 2>&1 &

# Run test split
CUDA_VISIBLE_DEVICES=1 python3 run_projections_mvtn_parallel_test_train.py \
    --nb_views $NB_VIEWS \
    --view_config $VIEW_CONFIG \
    --dir_output $DIR_OUTPUT \
    --split test \
    > logs/test_${VIEW_CONFIG}_${NB_VIEWS}-aligned.log 2>&1 &

# Wait for both jobs
wait

echo "✅ Both train and test projections completed."
