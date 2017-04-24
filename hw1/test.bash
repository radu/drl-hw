#!/bin/bash
set -eux
for e in Ant-v1 HalfCheetah-v1 Hopper-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
    python train_net.py $e $e --num_rollouts=5 --render
done
