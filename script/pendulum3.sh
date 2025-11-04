for friction in 0.4 ; do
    python3 /home/tedeschi_bpo/learn_RL/run_ns_retrain0.py \
        --dir /home/tedeschi_bpo/learn_RL/results/pendulum_friction/retrainfrom03/test{$friction} \
        --ite 200 --std 0.2 --env pendulum \
        --horizon 200 --batch 100 --gamma 0.99 \
        --n_trial 1 --pol deep_gaussian --n_jobs 27 \
        --lr 1e-3 --lr_strategy constant  --estimator GPOMDP \
        --verbose 1 --baseline peters --starting_seed 1 \
        --friction $friction
done