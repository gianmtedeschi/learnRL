#!/bin/bash

HORIZONS=(200 500 700)
PLANNING_HORIZONS=(1 2 3 5 10)
GAMMA=(0.999)

for G in "${GAMMA[@]}"
do
	for H in "${HORIZONS[@]}"
	do
    		for PH in "${PLANNING_HORIZONS[@]}"
    		do
            		python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
                		--dir /Users/gianmarcotedeschi/Projects/learnRL/results_swimmer_paper/ \
                		--ite 500 --std 1 --env swimmer \
                		--horizon $H --batch 100 --clip 0 --gamma $G \
                		--n_trial 20 --planning_horizon $PH --pol deep_gaussian --n_jobs 2 &
		done
		wait
	done
done

echo "Done"
