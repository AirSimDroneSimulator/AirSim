import tensorflow as tf
import numpy as np

from LinearEpsilonExplorer import LinearEpsilonExplorer
from ReplayMemory import ReplayMemory

class DQN_agent:

    def __init__(self, sess, 
                input_shape,
                action_num,
                lr=0.0025,
                gamma=0.9,
                explorer=LinearEpsilonExplorer(1, 0.1, 2000),
                minibatch=32,
                memory_size=2000,
                target_update_interval=500,
                train_after=200):
                
        self.sess = sess
        self.explorer = explorer
        self.minibatch = minibatch
        self.target_update_interval = target_update_interval
        self.train_after = train_after
        self.gamma = gamma
        self.input_shape = input_shape
        self.action_num = action_num
        
        self.replay_memory = ReplayMemory(memory_size)
        self.num_action_taken = 0
        
        self.X_Q = tf.placeholder(tf.float32, [None] + [self.input_shape])
        self.X_t = tf.placeholder(tf.float32, [None] + [self.input_shape])
        self.Q_network = self._build_network("Q_network", self.X_Q)
        self.target_network = self._build_network("target_network", self.X_t)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        with tf.variable_scope("optimizer"):
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
            # Q estimate
            actions_one_hot = tf.one_hot(self.actions, self.action_num)
            Q_pred = tf.reduce_sum(tf.multiply(self.Q_network, actions_one_hot), axis=1)
            # td_target
            self.td_target = tf.placeholder(tf.float32, [None])
            # loss
            self.loss = tf.losses.mean_squared_error(self.td_target, Q_pred)
            self.train_step = self.optimizer.minimize(self.loss)
        
    def _build_network(self, scope_name, X):
        
        with tf.variable_scope(scope_name):
        
            fc1 = tf.layers.dense(inputs=X, units=32, activation=tf.nn.relu)
            fc2 = tf.layers.dense(inputs=fc1, units=32, activation=tf.nn.relu)
            
            with tf.variable_scope("value"):
                value = tf.layers.dense(inputs=fc2, units=1)
                
            with tf.variable_scope("advantage"):
                advantage = tf.layers.dense(inputs=fc2, units=self.action_num)
            
            out = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return out
        
    def act(self, state):
        # gather random data before training
        # follow linearly decay epsilon greedy policy after training begin
        if self.num_action_taken >= self.train_after:
            
            if self.explorer.explore(self.num_action_taken - self.train_after):
                action = self.explorer.choose_random_action(self.action_num)
            else:
                env_history = state
                env_history = np.reshape(env_history, [1]+[self.input_shape])
                Q_values = self.sess.run(self.Q_network, feed_dict={self.X_Q : env_history})
                action = np.argmax(Q_values[0])
        else:
            action = self.explorer.choose_random_action(self.action_num)
        self.num_action_taken += 1
        
        return action
        
    def observe(self, pre_state, action, reward, post_state, terminal):
        # store transition in replay memory
        self.replay_memory.append(pre_state, action, reward, post_state, terminal)
        
    def train(self):
        # train the neural network after a certain number of actions
        # so that there are enough training samples in replay memory
        loss = 0
        if self.num_action_taken >= self.train_after:
            # retrieve data
            pre_states, actions, rewards, post_states, terminals = self.replay_memory.sample(self.minibatch)
            # Double DQN uses Q_network to choose action for post state
            # and then use target network to evaluate that policy
            Q_eval = self.sess.run(self.Q_network, feed_dict={self.X_Q:post_states})
            best_action = np.argmax(Q_eval, axis=1)
            # create one hot representation for action
            best_action_oh = np.zeros((best_action.size, self.action_num))
            best_action_oh[np.arange(best_action.size), best_action] = 1
            # evaluate through target_network
            Q_target = self.sess.run(self.target_network, feed_dict={self.X_t:post_states}) * best_action_oh
            Q_target = np.sum(Q_target, axis=1)
            y_batch = np.array(rewards) + self.gamma * Q_target * (1 - np.array(terminals))
            _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.X_Q:pre_states, self.actions:actions, self.td_target:y_batch})
        
        if self.num_action_taken % self.target_update_interval == 0:
            self.sess.run(self.update_target_net())
        
        return loss
    
    def update_target_net(self, dest_scope="target_network", src_scope="Q_network"):
        # return tf operations that copy weights to target network
        ops = []
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope)
        
        for src_var, dest_var in zip(src_vars, dest_vars):
            ops.append(dest_var.assign(src_var.value()))
        
        return ops