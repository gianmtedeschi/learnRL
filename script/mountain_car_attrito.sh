for friction in 0 0.1 0.2 0.3 0.4 0.5; do
  echo "Running with friction=$friction"

  python3 /home/tedeschi_bpo/learn_RL/run_ns.py \
    --dir /home/tedeschi_bpo/learn_RL/results/mountain_car_simm_v5/linear/attrito_naive/1e-3_rbf30/$friction-test \
    --ite 200 --std 0.7 --env mountain_car_5 \
    --horizon 999 --batch 100 --gamma 1 \
    --n_trial 1 --pol linear_gaussian --n_jobs 20 \
    --lr 5e-3 --lr_strategy adam --estimator GPOMDP \
    --verbose 1 --baseline peters --starting_seed 0 \
    --data_processor rbf_mcar_5 --algorithm on_policy\
    --friction $friction
done