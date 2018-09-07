

import numpy as np
import generate_data as gd
import tensorflow as tf
import time



class My_net(object):

	def __init__(self, params):

		self.SEED = int(params['seed'])
		self.BATCH_SIZE = int(params['batch_size'])
		self.EXEMPLAR_IMAGE_SIZE = int(params['exemplar_image_size'])
		self.SEARCH_IMAGE_SIZE = int(params['search_image_size'])
		self.NUM_CHANNELS = int(params['img_channels'])
		self.epsilon = float(params['batch_norm_eps'])
		self.ALL_PAIR = int(params['all_pair'])
		self.EPOC = int(params['epoc'])
		self.EVAL_NUM = int(params['eval_num'])
		self.learning_rate = float(params['learning_rate'])
		out1 = 96
		self.conv1_weights = tf.Variable(tf.truncated_normal([11, 11, 3, out1],stddev=0.1, seed=self.SEED,  dtype = tf.float32))
		self.conv1_biases = tf.Variable(tf.zeros([out1], dtype = tf.float32))

		self.scale1 = tf.Variable(tf.ones([out1]))
		self.beta1 = tf.Variable(tf.zeros([out1]))

		out2 = 256
		self.conv2_weights = tf.Variable(tf.truncated_normal([5, 5, out1/2, out2],stddev=0.1, seed=self.SEED,  dtype = tf.float32))
		self.conv2_biases = tf.Variable(tf.zeros([out2], dtype = tf.float32))

		self.scale2 = tf.Variable(tf.ones([out2]))
		self.beta2 = tf.Variable(tf.zeros([out2]))

		out3 = 384
		self.conv3_weights = tf.Variable(tf.truncated_normal([3, 3, out2, out3],stddev=0.1, seed=self.SEED,  dtype = tf.float32))
		self.conv3_biases = tf.Variable(tf.zeros([out3], dtype = tf.float32))

		self.scale3 = tf.Variable(tf.ones([out3]))
		self.beta3 = tf.Variable(tf.zeros([out3]))

		out4 = 384
		self.conv4_weights = tf.Variable(tf.truncated_normal([3, 3, out3/2, out4],stddev=0.1, seed=self.SEED,  dtype = tf.float32))
		self.conv4_biases = tf.Variable(tf.zeros([out4], dtype = tf.float32))

		self.scale4 = tf.Variable(tf.ones([out4]))
		self.beta4 = tf.Variable(tf.zeros([out4]))

		cout5 = 256
		self.conv5_weights = tf.Variable(tf.truncated_normal([3, 3, out4/2, cout5], stddev=0.1, seed=self.SEED,  dtype = tf.float32))
		self.conv5_biases = tf.Variable(tf.zeros([cout5], dtype = tf.float32))

		self.scale5 = tf.Variable(tf.ones([cout5]))
		self.beta5 = tf.Variable(tf.zeros([cout5]))

		self.exempler_train_data = tf.placeholder(tf.float32, shape = (self.BATCH_SIZE, self.EXEMPLAR_IMAGE_SIZE, self.EXEMPLAR_IMAGE_SIZE, self.NUM_CHANNELS))
		self.search_train_data = tf.placeholder(tf.float32, shape = (self.BATCH_SIZE, self.SEARCH_IMAGE_SIZE, self.SEARCH_IMAGE_SIZE, self.NUM_CHANNELS))
		self.train_label = tf.placeholder(tf.float32, shape = (self.BATCH_SIZE, 17, 17))
		self.weight_label = tf.placeholder(tf.float32, shape = (17, 17))

		self.loss, _ = self.produce_loss(self.BATCH_SIZE, self.exempler_train_data, self.search_train_data, self.train_label)
		self.optimizer = self.train_optimizer()


		self.exempler_eval_data = tf.placeholder(tf.float32, shape = (self.EVAL_NUM, self.EXEMPLAR_IMAGE_SIZE, self.EXEMPLAR_IMAGE_SIZE, self.NUM_CHANNELS))
		self.search_eval_data = tf.placeholder(tf.float32, shape = (self.EVAL_NUM, self.SEARCH_IMAGE_SIZE, self.SEARCH_IMAGE_SIZE, self.NUM_CHANNELS))
		self.eval_label = tf.placeholder(tf.float32, shape = (self.EVAL_NUM, 17, 17))

		self.eval_label = tf.placeholder(tf.float32, shape = (self.EVAL_NUM, 17, 17))
		_, self.eval_out = self.produce_loss(self.EVAL_NUM, self.exempler_eval_data, self.search_eval_data, self.eval_label)



		self.exempler_test_data = tf.placeholder(tf.float32, shape = (1, self.EXEMPLAR_IMAGE_SIZE, self.EXEMPLAR_IMAGE_SIZE, self.NUM_CHANNELS))
		self.search_test_data = tf.placeholder(tf.float32, shape = (1, self.SEARCH_IMAGE_SIZE, self.SEARCH_IMAGE_SIZE, self.NUM_CHANNELS))
		self.exempler_feature_outside = tf.placeholder(tf.float32, shape = (6, 6, 256))

		self.test_prediction = self.test_heat_map()
		self.out_feature = self.out_feature()




	def conv(self, input, kernel, biases, s_h, s_w,  padding="VALID", group=1):


		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	
		if group==1:
			conv = convolve(input, kernel)
		else:
			input_groups = tf.split(3, group, input)
			kernel_groups = tf.split(3, group, kernel)
			output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
			conv = tf.concat(3, output_groups)
		return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


	def model(self, images):


		conv1 = self.conv(images, self.conv1_weights, self.conv1_biases, 2, 2)
		batch_mean1, batch_var1 = tf.nn.moments(conv1,[0, 1,2])
		BN1 = tf.nn.batch_normalization(conv1,batch_mean1,batch_var1,self.beta1,self.scale1,self.epsilon)
		relu1 = tf.nn.relu(BN1)
		pool1 = tf.nn.max_pool(relu1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')


		conv2 = self.conv(pool1, self.conv2_weights, self.conv2_biases, 1, 1, group=2)
		batch_mean2, batch_var2 = tf.nn.moments(conv2,[0, 1,2])
		BN2 = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,self.beta2,self.scale2,self.epsilon)
		relu2 = tf.nn.relu(BN2)
		pool2 = tf.nn.max_pool(relu2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')


		conv3 = self.conv(pool2, self.conv3_weights, self.conv3_biases, 1, 1)
		batch_mean3, batch_var3 = tf.nn.moments(conv3,[0, 1,2])
		BN3 = tf.nn.batch_normalization(conv3,batch_mean3,batch_var3,self.beta3,self.scale3,self.epsilon)
		relu3 = tf.nn.relu(BN3)



		conv4 = self.conv(relu3, self.conv4_weights, self.conv4_biases, 1, 1, group=2)
		batch_mean4, batch_var4 = tf.nn.moments(conv4,[0, 1,2])
		BN4 = tf.nn.batch_normalization(conv4,batch_mean4,batch_var4,self.beta4,self.scale4,self.epsilon)
		relu4 = tf.nn.relu(BN4)



		conv5 = self.conv(relu4, self.conv5_weights, self.conv5_biases, 1, 1, group=2)
		batch_mean5, batch_var5 = tf.nn.moments(conv5,[0, 1,2])
		BN5 = tf.nn.batch_normalization(conv5,batch_mean5,batch_var5,self.beta5,self.scale5,self.epsilon)


		return BN5


	def produce_loss(self, BATCH_N, exempler_data, search_data, label):

		with tf.variable_scope("siamese") as scope:
			exemplar_out = self.model(exempler_data)
			scope.reuse_variables()
			search_out = self.model(search_data)

			exemplar_out_split = tf.unpack(exemplar_out)
			search_out_split = tf.unpack(search_out)


			map_split = []
			#0.001
			adjust_param = tf.constant(1, dtype = tf.float32)
			out_map = []
			for i in xrange(len(exemplar_out_split)):
				exemplar_map = tf.expand_dims(exemplar_out_split[i], -1)
				search_map = tf.expand_dims(search_out_split[i], 0)
				conv_out = tf.nn.conv2d(search_map, exemplar_map, strides = [1,1,1,1], padding = 'VALID')
				conv_out = tf.squeeze(conv_out)

				out_map.append(conv_out)

				conv_out = tf.scalar_mul(adjust_param, conv_out)
				conv_out = tf.mul(conv_out, self.weight_label)
				map_split.append(conv_out)

				# tmp_eval += tf.reduce_mean(tf.mul(conv_out, eval_final))
				# max_eval += tf.reduce_max(conv_out)
				# min_eval += tf.reduce_min(conv_out)



			add_1 = tf.ones([BATCH_N, 17, 17], tf.float32)
			compare_0 = tf.zeros([BATCH_N, 17, 17], tf.float32)
			tmp_label = tf.sub(compare_0, label)
			map_unit = tf.pack(map_split)
			tmp_conv = tf.sub(compare_0, tf.abs(map_unit))
			return tf.reduce_mean(tf.add(tf.maximum(tf.mul(map_unit, tmp_label), compare_0),  tf.log(tf.add(add_1, tf.exp(tmp_conv))))), out_map


	def train_optimizer(self):
		global_step = tf.Variable(0, dtype=tf.float32)
		learning_rate = tf.train.exponential_decay(
			self.learning_rate,                # Base learning rate.
			global_step,  # Current index into the dataset.
			self.EPOC/ 3 * self.ALL_PAIR/self.BATCH_SIZE,          # Decay step.
			0.1,                # Decay rate.
			staircase=True)


		optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=global_step)
		return optimizer

	def out_feature(self):
		exempler_feature = tf.squeeze(self.model(self.exempler_test_data))
		return exempler_feature

	def test_heat_map(self):
		search_feature = tf.squeeze(self.model(self.search_test_data))

		exemplar_map_test = tf.expand_dims(self.exempler_feature_outside, -1)
		search_map_test = tf.expand_dims(search_feature, 0)
		test_prediction = tf.squeeze(tf.nn.conv2d(search_map_test, exemplar_map_test, strides = [1,1,1,1], padding = 'VALID'))

		return test_prediction


