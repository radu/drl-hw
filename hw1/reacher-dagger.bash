#!/bin/bash
set -eux
for e in Reacher-v1
do
    python train_dagger.py $e $e experts/$e.pkl --num_rollouts=500 --retrain_steps=10 --training_iters=5 --batch_size=10 --n_hidden1=32 --n_hidden2=8 --render
done

