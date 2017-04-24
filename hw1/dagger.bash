#!/bin/bash
set -eux
for e in Reacher-v1
do
    python train_dagger.py $e $e experts/$e.pkl --num_rollouts=5 --render --batch_size=50 --training_iters=200 --n_hidden1=32 --n_hidden2=16 --beta=0.0001
done

for e in Walker2d-v1 Ant-v1 HalfCheetah-v1 Hopper-v1 Humanoid-v1 
do
    python train_dagger.py $e $e experts/$e.pkl --num_rollouts=5 --render
done

