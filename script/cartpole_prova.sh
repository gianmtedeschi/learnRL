#!/bin/bash
python3 /Users/sarazappia/Desktop/learnRL/run_ns.py \
    --dir /Users/sarazappia/Desktop/learnRL/results_bpo/test \
    --ite 300 --std 1 --env cartpole \
    --horizon 200 --batch 100 --gamma 1 \
    --n_trial 1 --pol deep_gaussian --n_jobs 8 \
    --lr 0.001 --lr_strategy constant --animate --estimator GPOMDP \
    --debug 
   