#!/bin/bash

LR=(0.1 0.01 0.001 0.05)
STD=(0.1 0.5 1 0.05)

for s in "${STD[@]}"
do
    for lr in "${LR[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/results_pendulum/ \
            --ite 500 --std $s --env pendulum \
            --horizon 200 --batch 100 --clip 0 --gamma 0.999 \
            --n_trial 1 --planning_horizon 1 --pol deep_gaussian --n_jobs 2 \
            --lr $lr --animate --lr_strategy adam
    done
    wait
done


echo "Done"



