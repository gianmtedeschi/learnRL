#!/bin/bash

HORIZONS=(100 500 1000)
PLANNING_HORIZONS=(1 3 10 20)

for H in "${HORIZONS[@]}"
do
    for PH in "${PLANNING_HORIZONS[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py --dir /Users/gianmarcotedeschi/Projects/learnRL/results/ --ite 500 --std 1 --env swimmer --horizon $H --batch 100 --clip 0 --n_trial 3 --planning_horizon $PH --pol deep_gaussian 
    done
done