#!/bin/bash

HORIZONS=(500)
PLANNING_HORIZONS_TEST=(1 3)
PLANNING_HORIZONS=(1 2 5 10)
BATCH=(5 10 20)

for H in "${HORIZONS[@]}"
do
    for PH in "${PLANNING_HORIZONS[@]}"
    do
        for B in "${BATCH[@]}"
        do
            python3 /Users/gianmarcotedeschi/Projects/learnRL/run_olop.py \
                --dir /Users/gianmarcotedeschi/Projects/learnRL/results/ \
                --ite 500 --std 1 --env swimmer \
                --horizon $H --batch $B --clip 0 \
                --n_trial 3 --planning_horizon $PH --pol deep_gaussian &
        done
    done
done

wait

echo "Done"