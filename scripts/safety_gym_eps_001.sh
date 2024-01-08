#!/bin/bash

# Recovery RL (model-free recovery)
export i=1
for i in {1..4}
do
	echo "RRL MF Run $i"
  python -m rrl_main --env-name safety_gym --use_recovery --MF_recovery --gamma_safe 0.8 --eps_safe 0.01 --logdir safety_gym_001 --logdir_suffix RRL_MF --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done
