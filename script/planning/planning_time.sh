#!/bin/bash

PLANNING_HORIZONS=(2 3 5 10)

for PH in "${PLANNING_HORIZONS[@]}"
do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/results_time/ \
            --ite 100 --std 0.5 --env dam --lr 5e-3 --lr_strategy adam --gamma 0.999 \
            --horizon 1825 --batch 100 --clip 0 --n_jobs 8 \
            --n_trial 3 --planning_horizon $PH --pol deep_gaussian  
done


echo "Done"