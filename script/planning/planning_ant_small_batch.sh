#!/bin/bash

HORIZONS=(200 500 700)
PLANNING_HORIZONS=(2 3 5 10)
GAMMA=(0.995 0.999)

for PH in "${PLANNING_HORIZONS[@]}"
do
    for H in "${HORIZONS[@]}"
    do
        for G in "${GAMMA[@]}"
        do
            python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
                        --dir /Users/gianmarcotedeschi/Projects/learnRL/results_ant/ \
                        --ite 2000 --std 0.1 --env ant \
                        --gamma $G --horizon $H --batch 100 --clip 0 --lr 0.001 --lr_strategy adam \
                        --n_trial 10 --planning_horizon $PH --pol deep_gaussian --n_jobs 4 &
        done
        wait
    done
done

echo "Done"