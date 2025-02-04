#!/bin/bash

HORIZONS=(200)
PLANNING_HORIZONS=(1 2 3 5 10)

for H in "${HORIZONS[@]}"
do
    for PH in "${PLANNING_HORIZONS[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/result_debug/ \
            --ite 200 --std 0.5 --env cartpole \
            --horizon $H --batch 100 --clip 0 --gamma 0.998 \
            --n_trial 3 --planning_horizon $PH --pol deep_gaussian --n_jobs 5 \
            --lr 0.001 --lr_strategy adam
    done
done

wait

echo "Done"



