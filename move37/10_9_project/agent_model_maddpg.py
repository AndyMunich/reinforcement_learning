import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

class MADDPG():
    def __init__(self, name, layer_norm=True, nb_actions=2, nb_input=16, nb_other_actions=4):
        gamma = 0.999
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_inp = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_inp = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_inp = tf.placeholder(shape=[None, nb_other_actions], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        def actor_network(name):
            with tf.variable_scope(name) as scope:
                x = state_inp
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-1e-3, maxval=1e-3))
                x = tf.nn.tanh(x)
            return x

        def critic_network(name, action_inp, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()

                x = state_inp
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action_inp], axis=-1)
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-1e-3, maxval=1e-3))
            return x

        self.action_output = actor_network(name + '_actor')
        self.critic_output = critic_network(name + '_critic', action_inp=tf.concat([action_inp, other_action_inp], axis=1))
        self.state_inp = state_inp
        self.action_inp = action_inp
        self.other_action_inp = other_action_inp
        self.reward = reward

        self.actor_optimizer = tf.train.AdamOptimizer(1e-4)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)

        self.actor_loss = -tf.reduce_mean(critic_network(name + '_critic', action_inp=tf.concat([self.action_output, other_action_inp], axis=1), reuse=True))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

    def train_actor(self, state, other_action, sess):
        sess.run(self.actor_train, {self.state_inp: state, self.other_action_inp: other_action})

    def train_critic(self, state, action, other_action, target, sess):
        sess.run(self.critic_train, {self.state_inp: state, self.action_inp: action, self.other_action_inp: other_action, self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_inp: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output, {self.state_inp: state, self.action_inp: action, self.other_action_inp: other_action})
