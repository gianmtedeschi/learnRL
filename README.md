## Set up the environment.
You need an `anaconda3` environment with python 3.9
```bash
conda create --name name python=3.9
conda activate name
```

Install the packages.
```bash
pip3 install -r requirements.txt
```

## Run experiments.
All you need is in `run.py`, which requires several parameters:
- "--dir": specifies the directory in which will be saved the results;
- "--ite": how many iterations the algorithm must do;
- "--alg": the algorithm to run, you can select "pg" or "split";
- "--estimator": specifies which estimator to use;
- "--std": the exploration amount, it is $\sigma^2$;
- "--pol": the policy to use, you can select "gaussian" or "split_gaussian";
- "--env": the environment on which the learning has to be done, you can select "swimmer", "half_cheetah", "ant", "lq", "minigolf";
- "--horizon": set the horizon of the problem;
- "--gamma": set the discount factor of the problem;
- "--lr": set the step size;
- "--lr_strategy": set the learning rate schedule, you can select "constant" or "adam";
- "--batch": specifies how many trajectories are evaluated in each iteration;
- "--clip": specifies whether to apply action clipping, you can select "0" or "1";
- "--n_trials": specifies how many run of the same experiments has to be done.
- "--verbose": print debug information.
- "--baseline": specifies which baselinse adopt
- "--n_jobs": specifies how many parallel jobs to use.

Only for the GAPS algorithm:
- "--alpha": specifies the alpha parameter for the split check criteria;
- "--max_splits": specifies the maximum number of split that can be performed;

Only for the LQR environment:
- "--lq_state_dim": specifies the state dimension for the LQR environment;
- "--lq_action_dim": specifies the action dimension for the LQR environment;


Here is an example running PG on Swimmer:
```bash
python3 run.py --dir /your/path --alg pg --ite 100 --std 1 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.1 --lr_strategy adam --clip 1 --batch 30 --n_trials 1
```