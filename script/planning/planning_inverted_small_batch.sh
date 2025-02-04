#!/bin/bash

HORIZONS=(200)
PLANNING_HORIZONS=(1 2 3 5 10)
GAMMA=(0.995 0.999)

for G in "${GAMMA[@]}"
do
    for H in "${HORIZONS[@]}"
    do
        for PH in "${PLANNING_HORIZONS[@]}"
        do
                python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
                    --dir /Users/gianmarcotedeschi/Projects/learnRL/results_inv/ \
                    --ite 200 --std 0.5 --env inverted_pendulum \
                    --gamma $G --horizon $H --batch 10 --clip 0 --lr 0.01 --lr_strategy adam \
                    --n_trial 5 --planning_horizon $PH --pol deep_gaussian --n_jobs 2 &
        done
        wait
    done
done



echo "Done"