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
import pudb

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
    inds = np.random.randint(obs.shape[0], size=batch_size)
    return (obs[inds,:], act[inds,:])

    #start = n*batch_size
    #end = start + batch_size
    #if obs.shape[0] < start or act.shape[0] < start:
    #    return None
    #elif obs.shape[0] < start or act.shape[0] < start:
    #    return (obs[start:], act[start:])
    #else:
    #    return (obs[start:end], act[start:end])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('policy', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=100000)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of policy roll outs')

    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--training_iters', type=int, default = 500)
    parser.add_argument('--display_step', type=int, default = 20)

    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--render_epoch_start', type=int, default=10)

    parser.add_argument('--beta', type=float, default = 0.001)

    parser.add_argument('--n_hidden1', type=int, default = 500)
    parser.add_argument('--n_hidden2', type=int, default = 64)

    parser.add_argument('--retrain_steps', type=int, default=100)

    args = parser.parse_args()

    retrain_steps = args.retrain_steps

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_iters = args.training_iters
    display_step = args.display_step

    num_epochs = args.num_epochs

    beta = args.beta

    n_hidden1 = args.n_hidden1
    n_hidden2 = args.n_hidden2

    obs_data, act_data = load_run(args.filename)
    obs_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]

    policy_fn = load_policy.load_policy(args.policy)

    n_input = obs_dim
    n_classes = act_dim

    x = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    keep_prob = tf.placeholder(tf.float32)

    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.5)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.5)),
        'out': tf.Variable(tf.truncated_normal([n_hidden2, n_classes], stddev=0.5)),
        'hout': tf.Variable(tf.truncated_normal([n_hidden1, n_classes], stddev=0.5)),
        'lin': tf.Variable(tf.truncated_normal([n_input, n_classes], stddev=0.5))
    }

    biases = {
        'h1': tf.Variable(tf.constant(0.1,shape=[n_hidden1])),
        'h2': tf.Variable(tf.constant(0.1,shape=[n_hidden2])),
        'out': tf.Variable(tf.constant(0.1,shape=[n_classes])),
        'lin': tf.Variable(tf.constant(0.1,shape=[n_classes]))
    }

    def net(x,weights, biases):
        h1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
        h1_drop = tf.nn.dropout(h1, keep_prob)
        h1_relu = tf.nn.tanh(h1_drop)
        lin = tf.add(tf.matmul(x, weights['lin']), biases['lin'])
        lin_drop = tf.nn.dropout(lin, keep_prob)
        net_out = tf.add(tf.matmul(h1_drop, weights['hout']),biases['out'])

        result = tf.add(net_out, lin)

        return result

    y_conv = net(x, weights, biases)

    cross_entropy = tf.losses.mean_squared_error(y_, y_conv)

    #cross_entropy = - tf.reduce_mean(tf.abs(y_- y_conv))

    regularizer = tf.nn.l2_loss(weights['h1'], 'h1_l2') + tf.nn.l2_loss(weights['h2'], 'h2_l2') + tf.nn.l2_loss(weights['out'], 'out_l2')

    loss = tf.reduce_mean(cross_entropy + beta * regularizer, name='xentropy_mean')

    tf.summary.scalar('loss', loss)

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    def train_net(sess, obs_data, act_data, batch_size, num_epochs=1):
        step = 0
        stop = False

#       while not stop:
        for step in range(num_epochs):
            (batch_x, batch_y) = next_batch(obs_data, act_data, step, batch_size)

            start_time = time.time()
            _, loss_value = sess.run([train_step, loss],
                                        feed_dict={x:batch_x, y_: batch_y,
                                                keep_prob:0.99})
            duration = time.time() - start_time

            if step % display_step == 0:
                print('step %d: loss %.2f (%.3f sec)' % (step, loss_value, duration))

#            step += 1
#            n = next_batch(obs_data, act_data, batch_size)

#            if n == None:
#                stop = True
#            else:
#                (batch_x,batch_y) = n

    with tf.Session() as sess:
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)


        for it in range(training_iters):

            print ("Iter %d"%it)
            train_net(sess, obs_data, act_data, batch_size)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        new_obs = np.array([])
        new_act = np.array([])

        for i in range(args.num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = y_conv.eval(feed_dict={x:obs[None,:], keep_prob:1.0})
                pol_action = policy_fn(obs[None,:])
                new_obs = np.append(new_obs, obs[None,:], axis=0) if new_obs.size else np.array(obs[None,:])
                new_act = np.append(new_act, pol_action, axis=0) if new_act.size else np.array(pol_action)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render and i > args.render_epoch_start:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
#               if steps % retrain_steps == 0:
#                   train_net(sess, new_obs, new_act, retrain_steps)
                if steps >= max_steps:
                    break
            #train_net(sess, new_obs, new_act)
            train_net(sess, new_obs, new_act, batch_size, num_epochs)
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        imitator_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}


if __name__ == '__main__':
    main()
