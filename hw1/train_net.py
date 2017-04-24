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

def next_batch(obs, act, n, batch_size):
    start = n*batch_size
    end = start + batch_size
    if obs.shape[0] < end or act.shape[0] < end:
        return None
    else:
        return (obs[start:end], act[start:end])

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

    learning_rate = 0.005
    batch_size = 1000
    training_iters = 100
    display_step = 20

    beta = 0.001

    n_input = obs_dim
    n_hidden1 = 128
    n_hidden2 = 64
    n_classes = act_dim

    x = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    keep_prob = tf.placeholder(tf.float32)

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
        'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
        'out': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
    }

    biases = {
        'h1': tf.Variable(tf.constant(0.1,shape=[n_hidden1])),
        'h2': tf.Variable(tf.constant(0.1,shape=[n_hidden2])),
        'out': tf.Variable(tf.constant(0.1,shape=[n_classes]))
    }

    def net(x,weights, biases):
        h1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
        #h1_drop = tf.nn.dropout(h1, keep_prob)
        h1_relu = tf.nn.tanh(h1)
        h2 = tf.add(tf.matmul(h1_relu, weights['h2']), biases['h2'])
        #h2_drop = tf.nn.dropout(h2, keep_prob)
        h2_relu = tf.nn.tanh(h2)
        result = tf.add(tf.matmul(h2_relu, weights['out']),biases['out'])

        return result

    y_conv = net(x, weights, biases)

    cross_entropy = tf.losses.absolute_difference(y_, y_conv)

    #cross_entropy = - tf.reduce_mean(tf.abs(y_- y_conv))

    regularizer = tf.nn.l2_loss(weights['h1'], 'h1_l2') + tf.nn.l2_loss(weights['h2'], 'h2_l2') + tf.nn.l2_loss(weights['out'], 'out_l2')

    loss = tf.reduce_mean(cross_entropy + beta * regularizer, name='xentropy_mean')

    tf.summary.scalar('loss', loss)

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)


        for it in range(training_iters):
            step = 0
            stop = False

            (batch_x, batch_y) = next_batch(obs_data, act_data, step, batch_size)

            while not stop:
                start_time = time.time()
                _, loss_value = sess.run([train_step, loss],
                                         feed_dict={x:batch_x, y_: batch_y,
                                                    keep_prob:0.99})
                duration = time.time() - start_time

                if step % display_step == 0:
                    print('Iter %d, step %d: loss %.2f (%.3f sec)' % (it, step, loss_value, duration))

                step += 1
                n = next_batch(obs_data, act_data, step, batch_size)

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
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = y_conv.eval(feed_dict={x:obs[None,:], keep_prob:1.0})
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
