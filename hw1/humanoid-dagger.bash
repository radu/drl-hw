#!/bin/bash
set -eux
for e in Humanoid-v1
do
    python train_dagger.py $e $e experts/$e.pkl --num_rollouts=5000 --render_epoch_start=2000 --n_hidden1=500 --batch_size=2000 --beta=0.001 --training_iters=20000 --learning_rate=0.001 --retrain_steps=20 --render
done

