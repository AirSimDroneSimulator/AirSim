import tensorflow as tf
import numpy as np

class Critic:
	def __init__(self, sess, state_shape, action_dim, minibatch_size, lr=1e-3, tau=0.001):
		self.sess = sess
		self.tau = tau
		self.minibatch_size = minibatch_size
		
		self.reward = tf.placeholder(tf.float32, [None, 1])
		self.td_target = tf.placeholder(tf.float32, [None, 1])
		
		# input for Q network
		self.state = tf.placeholder(tf.float32, [None, state_shape])
		self.img = tf.placeholder(tf.float32, [None, 64, 64, 1])
		self.action = tf.placeholder(tf.float32, [None, action_dim])
		
		#input for target network
		self.t_state = tf.placeholder(tf.float32, [None, state_shape])
		self.t_img = tf.placeholder(tf.float32, [None, 64, 64, 1])
		self.t_action = tf.placeholder(tf.float32, [None, action_dim])
		
		with tf.variable_scope("critic"):
			self.eval_net = self._build_network(self.state, self.action, self.img, "eval_net")
			self.target_net = self._build_network(self.t_state, self.t_action, self.t_img, "target_net")
		
		self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic/eval_net")
		self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic/target_net")
		
		self.loss = tf.losses.mean_squared_error(self.td_target, self.eval_net)
		self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
		self.action_gradients = tf.gradients(self.eval_net, self.action)
		
		self.update_ops = self._update_target_net_op()
		
	def _build_network(self, X, action, image, scope):
		with tf.variable_scope(scope):
			init_w1 = tf.truncated_normal_initializer(0., 3e-4)
			init_w2 = tf.random_uniform_initializer(-0.05, 0.05)

			conv1 = tf.layers.conv2d(image, 32, [3,3], strides=[4,4], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
			conv2 = tf.layers.conv2d(conv1, 32, [3,3], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
			conv3 = tf.layers.conv2d(conv2, 32, [3,3], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
			flatten = tf.layers.flatten(conv3) # shape(None, 4*4*32)
			concat = tf.concat([flatten, action, X], 1)

			fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
			fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
			Q = tf.layers.dense(inputs=fc2, units=1, kernel_initializer=init_w2)
		return Q
		
	def target_net_eval(self, states, actions):
		imgs, dstates = self._seperate_image(states)
		Q_target = self.sess.run(self.target_net, feed_dict={self.t_state:dstates, self.t_action:actions, self.t_img:imgs})
		return Q_target
		
	def action_gradient(self, states, actions):
		imgs, dstates = self._seperate_image(states)
		return self.sess.run(self.action_gradients, feed_dict={self.state:dstates, self.action:actions, self.img:imgs})[0]
		
	def train(self, states, actions, td_target):
		imgs, dstates = self._seperate_image(states)
		actions = actions.reshape([self.minibatch_size,1])
		feed_dict = {self.state:dstates, self.action:actions, self.td_target:td_target, self.img:imgs}
		self.sess.run(self.train_step, feed_dict=feed_dict)
		
	def _update_target_net_op(self):
		ops = [tf.assign(dest_var, (1-self.tau) * dest_var + self.tau * src_var)
			   for dest_var, src_var in zip(self.target_param, self.eval_param)]
		return ops

	def _seperate_image(self, states):
		images = np.array([state[0] for state in states])
		dstates = np.array([state[1] for state in states])
		return images, dstates