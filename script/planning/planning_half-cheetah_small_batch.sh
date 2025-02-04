#!/bin/bash

HORIZONS=(200)
PLANNING_HORIZONS=(1 2 3 5 10)

for H in "${HORIZONS[@]}"
do
    for PH in "${PLANNING_HORIZONS[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/results_hc/ \
            --ite 500 --env half_cheetah \
            --horizon $H --batch 10 --gamma 0.999 --lr_strategy adam --clip 0 \
            --n_trial 10 --planning_horizon $PH --pol deep_gaussian --lr 0.001  --n_jobs 2 & 
    done
    wait
done

echo "Done"