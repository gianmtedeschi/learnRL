#!/bin/bash

HORIZONS=(200)
PLANNING_HORIZONS=(1 2 3 5 10)

for H in "${HORIZONS[@]}"
do
    for PH in "${PLANNING_HORIZONS[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/result_debug/ \
            --ite 200 --std 0.05 --env river \
            --horizon $H --batch 50 --clip 0 --gamma 0.998 \
            --n_trial 20 --planning_horizon $PH --pol deep_gaussian --n_jobs 2 --lr 0.05 &
    done
done

wait

echo "Done"