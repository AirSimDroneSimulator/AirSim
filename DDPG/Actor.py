import tensorflow as tf
import numpy as np

class Actor:
	def __init__(self, sess, action_bound, action_dim, state_shape, lr=1e-4, tau=0.001):
		self.sess = sess
		self.action_bound = action_bound
		self.action_dim = action_dim
		self.state_shape = state_shape
		self.tau = tau
		
		self.state = tf.placeholder(tf.float32, [None, state_shape])
		self.img = tf.placeholder(tf.float32, [None, 64, 64, 1])
		self.post_state = tf.placeholder(tf.float32, [None, state_shape])
		self.post_img = tf.placeholder(tf.float32, [None, 64, 64, 1])
		self.Q_gradient =  tf.placeholder(tf.float32, [None, action_dim])
		
		with tf.variable_scope("actor"):
			self.eval_net = self._build_network(self.state, self.img, "eval_net")
			# target net is used to predict action for critic
			self.target_net = self._build_network(self.post_state, self.post_img, "target_net")
		
		self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor/eval_net")
		self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor/target_net")
		
		# use negative Q gradient to guide gradient ascent
		self.policy_gradient = tf.gradients(ys=self.eval_net, xs=self.eval_param, grad_ys=-self.Q_gradient)
		self.train_step = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.policy_gradient, self.eval_param))
		
		self.update_ops = self._update_target_net_op()
		
	def _build_network(self, X, image, scope):
		with tf.variable_scope(scope):
			init_w1 = tf.truncated_normal_initializer(0., 3e-4)
			init_w2 = tf.random_uniform_initializer(-0.05, 0.05)

			conv1 = tf.layers.conv2d(image, 32, [5,5], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
			conv2 = tf.layers.conv2d(conv1, 32, [5,5], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
			conv3 = tf.layers.conv2d(conv2, 32, [5,5], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
			conv4 = tf.layers.conv2d(conv3, 32, [5,5], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
			flatten = tf.layers.flatten(conv4) # shape(None, 4*4*32)
			concat = tf.concat([flatten, X], 1)

			fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
			fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
			fc3 = tf.layers.dense(inputs=fc2, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
			action_normal = tf.layers.dense(inputs=fc3, units=self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w2)
			action = tf.multiply(action_normal, self.action_bound)
		return action
		
	def act(self, state):
		img, dstate = state
		img = np.reshape(img, [1, 64, 64, 1])
		dstate = np.reshape(dstate, [1, self.state_shape])
		action = self.sess.run(self.eval_net, feed_dict={self.state:dstate, self.img:img})[0]
		return action
		
	def predict_action(self, states):
		imgs, dstates = self._seperate_image(states)
		pred_actions = self.sess.run(self.eval_net, feed_dict={self.state:dstates, self.img:imgs})
		return pred_actions
		
	def target_action(self, post_states):
		imgs, dstates = self._seperate_image(post_states)
		actions = self.sess.run(self.target_net, feed_dict={self.post_state:dstates, self.post_img:imgs})
		return actions
		
	def train(self, Q_gradient, states):
		imgs, dstates = self._seperate_image(states)
		self.sess.run(self.train_step, feed_dict={self.state:dstates, self.img:imgs, self.Q_gradient:Q_gradient})
		
	def _update_target_net_op(self):
		ops = [tf.assign(dest_var, (1-self.tau) * dest_var + self.tau * src_var)
			   for dest_var, src_var in zip(self.target_param, self.eval_param)]
		return ops

	def _seperate_image(self, states):
		images = np.array([state[0] for state in states])
		dstates = np.array([state[1] for state in states])
		return images, dstates