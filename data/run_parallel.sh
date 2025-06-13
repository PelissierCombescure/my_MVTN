#!/bin/bash

NB_VIEWS=$1
VIEW_CONFIG=$2
DIR_OUTPUT=$3
N_JOBS=$4

# Compute total length of both datasets
TOTAL=$(python3 -c "
from mvtorch.data import ModelNet40;
t1 = len(ModelNet40(data_dir='/home/mpelissi/Dataset/ModelNet40/', split='train'))
t2 = len(ModelNet40(data_dir='/home/mpelissi/Dataset/ModelNet40/', split='test'))
print(t1 + t2)
")

CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))
mkdir -p logs

for ((i=0; i<N_JOBS; i++)); do
    START=$((i * CHUNK))
    END=$(((i + 1) * CHUNK))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

    CUDA_VISIBLE_DEVICES=$((i % 2)) python3 run_projections_mvtn_parallel.py \
        --nb_views $NB_VIEWS \
        --view_config $VIEW_CONFIG \
        --dir_output $DIR_OUTPUT \
        --start_idx $START \
        --end_idx $END \
        > logs/job_$i.log 2>&1 &
done

wait
echo "✅ All $N_JOBS parallel jobs done"
