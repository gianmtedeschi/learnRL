#!/bin/bash

# HORIZONS=(200)
PLANNING_HORIZONS=(1 2 3 5 10)
GAMMA=(0.995 0.999)

for PH in "${PLANNING_HORIZONS[@]}"
do
    python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
        --dir /Users/gianmarcotedeschi/Projects/learnRL/result_dam/ \
        --ite 100 --std 0.5 --env dam \
        --horizon 1825 --batch 10 --clip 0 --gamma 0.999 \
        --n_trial 1 --planning_horizon $PH --pol deep_gaussian --n_jobs 22 --lr 5e-3 --lr_strategy adam &
done
wait


echo "Done"
