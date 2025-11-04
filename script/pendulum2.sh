#!/bin/bash

for friction in 0.3 0.4 ; do
    python3 /home/tedeschi_bpo/learn_RL/run_ns.py \
        --dir /home/tedeschi_bpo/learn_RL/results/pendulum_friction/decay/test{$friction} \
        --ite 200 --std 1 --env pendulum \
        --horizon 200 --batch 100 --gamma 0.99 \
        --n_trial 1 --pol deep_gaussian --n_jobs 26 \
        --lr 1e-3 --lr_strategy constant  --estimator GPOMDP \
        --verbose 1 --baseline peters --starting_seed 1 \
        --friction $friction
done