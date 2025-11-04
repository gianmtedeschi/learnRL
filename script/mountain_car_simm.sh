#!/bin/bash

echo "Starting hyperparameter sweep (std=0.75 fixed)..."

for lr in  1e-2 5e-3 1e-3; do
  for layer in 64; do
    for batch in 200; do

      echo "Running lr=$lr, layer=$layer, batch=$batch, std=0.75"

      python3 /home/tedeschi_bpo/learn_RL/run_ns.py \
        --dir /home/tedeschi_bpo/learn_RL/results/mountain_car_simm_v3/deep_bpo_xavier_initialization/ultimo_tentativo/_loop_lr${lr}_lay${layer}_b${batch} \
        --ite 200 \
        --std 0.75 \
        --env mountain_car_simmetric \
        --horizon 999 \
        --batch $batch \
        --gamma 1 \
        --n_trial 1 \
        --pol deep_gaussian \
        --n_jobs 50 \
        --lr $lr \
        --lr_strategy adam \
        --estimator GPOMDP \
        --verbose 1 \
        --baseline peters \
        --starting_seed 0 \
        --data_processor identity \
        --animate \
        --behavioural_std 0.6 \
        --algorithm on_policy \
        --layers $layer

    done
  done
done

echo "All runs completed!"


