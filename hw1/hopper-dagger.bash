#!/bin/bash
set -eux
for e in Hopper-v1
do
    python train_dagger.py $e $e experts/$e.pkl --num_rollouts=100 --retrain_steps=10 --training_iters=3 --batch_size=200 --render
done

