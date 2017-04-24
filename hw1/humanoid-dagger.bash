#!/bin/bash
set -eux
for e in Humanoid-v1
do
    python train_dagger.py $e $e experts/$e.pkl --num_rollouts=50 --n_hidden1=1024 --n_hidden2=512 --batch_size=10 --training_iters=50 --retrain_steps=5 --render
done

