#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import time

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def load_run(filename):
    expert_data = pickle.load(open("%s.pkl"%(filename),"rb"))

    obs = expert_data['observations']
    act = expert_data['actions']

    return obs, act[:,0,:]

def next_batch(obs, act, n, batch_size, n_steps):
    start = n*batch_size
    end = start + batch_size
    if obs.shape[0] < end or act.shape[0] < end:
        print('%d or %d too small for %d'%(obs.shape[0],act.shape[0], end))
        return None
    else:
        obs_arr = []
        for i in range(batch_size):
            cur_step = n * (batch_size-1) + i
            obs_arr.append(obs[cur_step - n_steps:cur_step])
        return (obs_arr, act[start:end])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of policy roll outs')
    args = parser.parse_args()

    obs_data, act_data = load_run(args.filename)
    obs_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]

    learning_rate = 0.001
    training_iters = 200
    batch_size = 100
    display_step = 10

    n_input = obs_dim
    n_hidden = 128
    n_classes = act_dim

    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def net(x,weights, biases):

        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=act_dim)]

        lstm_cell = tf.contrib.learn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    y_conv = net(x, weights, biases)

    cross_entropy = tf.losses.mean_pairwise_squared_error(y_, y_conv)

    #cross_entropy = - tf.reduce_mean(tf.abs(y_- y_conv))
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    tf.summary.scalar('loss', loss)

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)

        step = 1
        stop = False

        (batch_x, batch_y) = next_batch(obs_data, act_data, step, batch_size, n_steps)

        while not stop:
            start_time = time.time()
            _, loss_value = sess.run([train_step, loss], feed_dict={x:batch_x, y_: batch_y})
            duration = time.time() - start_time

            if step % 10 == 0:
                print('Step %d: loss %.2f (%.3f sec)' % (step, loss_value, duration))

            step += 1
            n = next_batch(obs_data, act_data, step, batch_size, n_steps)

            if n == None:
                stop = True
            else:
                (batch_x,batch_y) = n

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                print(obs.shape)
                action = y_conv.eval(feed_dict={x:obs[None,:]})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        imitator_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}


if __name__ == '__main__':
    main()
