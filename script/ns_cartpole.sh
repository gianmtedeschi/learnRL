#!/bin/bash

FORCE=(10)
# FRICTION=(0 0.01)
FRICTION=(0.01)

for F in "${FORCE[@]}"
do
    for FR in "${FRICTION[@]}"
    do
        python3 /Users/gianmarcotedeschi/Projects/learnRL/run_ns.py \
            --dir /Users/gianmarcotedeschi/Projects/learnRL/results_debug/ \
            --ite 1 --std 1 --env pendulum \
            --horizon 1000 --batch 100 --clip 0 --gamma 1 \
            --n_trial 1 --pol deep_gaussian --n_jobs 8 \
            --lr 0 --lr_strategy adam \
            --force $F --friction $FR --animate
    done
    wait
done


echo "Done"



