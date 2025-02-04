#!/bin/bash

HORIZONS=(100 500)
PLANNING_HORIZONS=(1 2 3 5 10)

for H in "${HORIZONS[@]}"
do
    for PH in "${PLANNING_HORIZONS[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/results_hopper/ \
            --ite 500 --env hopper \
            --horizon $H --batch 10 --gamma 0.998 --lr_strategy adam --clip 0 \
            --n_trial 10 --planning_horizon $PH --pol deep_gaussian --n_jobs 2 --std 1 &
    done
    wait
done

echo "Done"