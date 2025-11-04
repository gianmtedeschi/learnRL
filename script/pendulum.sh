#!/bin/bash

for kl in 0 ; do
    for behavioural_std in 0.75 ; do
        python3 /home/tedeschi_bpo/learn_RL/run_ns.py \
            --dir /home/tedeschi_bpo/learn_RL/results/pendulum_friction/prova_like_lifelong/p0185std{$behavioural_std}kl{$kl} \
            --ite 200 --std $behavioural_std --env pendulum \
            --horizon 200 --batch 100 --gamma 0.99 \
            --n_trial 1 --pol deep_gaussian --n_jobs 20 \
            --lr 1e-3 --lr_strategy constant  --estimator GPOMDP \
            --verbose 1 --baseline peters --starting_seed 1 \
            --friction 0.3 --kl $kl --behavioural_std $behavioural_std --defensive_batchsize 25
    done
done