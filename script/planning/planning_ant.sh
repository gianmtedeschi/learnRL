#!/bin/bash

HORIZONS=(500)
PLANNING_HORIZONS=(1 2 5 10)

for H in "${HORIZONS[@]}"
do
    for PH in "${PLANNING_HORIZONS[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/results_planning/ \
            --ite 500 --env ant \
            --horizon 500 --batch 100 --gamma 0.998 --lr_strategy adam --lr 0.001 --std 1 --clip 0 \
            --n_trial 3 --planning_horizon $1 --pol deep_gaussian --n_jobs 8 &
    done
done

wait

echo "Done"