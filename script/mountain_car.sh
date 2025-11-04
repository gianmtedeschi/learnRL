#!/bin/bash

for std in 0.75; do
  echo "Running with std=$std"

  python3 /home/tedeschi_bpo/learn_RL/run_ns.py \
    --dir /home/tedeschi_bpo/learn_RL/results/mountain_car/deep_bpo_xavier_initialization/lr_5e-3__behavlr_1e-4_it500_strategy_adam_std_${std}_fixed_baseline_v2/test \
    --ite 200 --std $std --env mountain_car \
    --horizon 999 --batch 100 --gamma 1 \
    --n_trial 5 --pol deep_gaussian --n_jobs 20 \
    --lr 5e-3 --lr_strategy adam --estimator GPOMDP \
    --verbose 1 --baseline peters --starting_seed 1 \
    --data_processor identity --animate \
    --behavioural_std $std
done
