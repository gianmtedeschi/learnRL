#!/bin/bash

python3 /home/tedeschi_bpo/learn_RL/run_ns.py \
    --dir /home/tedeschi_bpo/learn_RL/results/mountain_car_simm_v3/deep_bpo_xavier_initialization/montagnette/ \
    --ite 1 --std 0.7 --env mountain_car_4 --horizon 999 --batch 100 --gamma 1 --n_trial 1 \
    --pol linear_gaussian --n_jobs 5 --lr 1e-3 --lr_strategy adam --estimator GPOMDP --verbose 1 \
    --baseline peters --starting_seed 0 --data_processor rbf_mcar \
    --animate --algorithm on_policy \