#!/bin/bash
set -eux
for e in Humanoid-v1
do
    python run_expert.py experts/$e.pkl $e --num_rollouts=2 --max_timesteps=100
done
