#!/bin/bash

STATE=(3)
ACTION=(3)
PLANNING_HORIZONS=(1 2 3 5 10)
NOISE=(0 0.2 0.4 0.6 0.8)

for N in "${NOISE[@]}"
do
    for S in "${STATE[@]}"
    do
        for A in "${ACTION[@]}"
        do
        for PH in "${PLANNING_HORIZONS[@]}"
            do
                python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
                    --dir /Users/gianmarcotedeschi/Projects/learnRL/results_lq/ \
                    --ite 250 --std 0.5 --env lq --gamma 0.999 \
                    --horizon 50 --batch 100 --clip 0 \
                    --n_trial 20 --planning_horizon $PH \
                    --pol deep_gaussian --lr_strategy adam --lr 0.005 --lq_action_dim $A --lq_state_dim $S \
                    --noise $N --n_jobs 2 &
            done
            wait
        done
    done
done


echo "Done"