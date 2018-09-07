import my_net as mn
import numpy as np
import generate_data as gd
import tensorflow as tf
import time
import cv2
import os

# rgbMean_z = [117.9743, 114.0875, 100.3012]
# rgbVariance_z = [[0.6832, 1.9544, 5.2611], [-0.9529, 0.6693, 5.3551], [0.2754, -2.5323, 5.4760]]
# rgbMean_x = [118.2404, 115.7477, 102.8751]
# rgbVariance_x = [[0.6318, 1.9295, 4.5166], [-0.8720, 0.7176, 4.6429], [0.2426, -2.4452, 4.9265]]


def prodect_label(batch_n):

	label = np.ones((batch_n, 17, 17), dtype = np.float32) * -1
	for k in xrange(my_net.BATCH_SIZE):
		for i in xrange(17):
			 for j in xrange(17):
				if i >= 6 and i <= 10 and j == 8:
					label[k][i][j] = 1
				if i >= 7 and i <= 9 and (j == 7 or j == 9):
					label[k][i][j] = 1
				if i == 8 and (j == 6 or j == 10):
					label[k][i][j] = 1


	label =  np.array(label)
	return label


def prodect_instance_weight():

	label = np.ones((17, 17), dtype = np.float32) * 0.0024
	for i in xrange(17):
		 for j in xrange(17):
			if i >= 6 and i <= 10 and j == 8:
				label[i][j] = 0.0385
			if i >= 7 and i <= 9 and (j == 7 or j == 9):
				label[i][j] = 0.0385
			if i == 8 and (j == 6 or j == 10):
				label[i][j] = 0.0385


	label =  np.array(label)
	return label






if __name__ == '__main__':

	params = {'seed' : 66478, 'batch_size' : 64, 'exemplar_image_size' : 127, 
	'search_image_size' : 255, 'img_channels' : 3, 'batch_norm_eps' : 1e-3, 
	'all_pair' : 10000, 'epoc' : 5, 'eval_num' : 64, 'learning_rate' : 0.001}
	my_net = mn.My_net(params)

	out_root = './t1/'


	if not os.path.exists(out_root): 
		os.mkdir(out_root)
	if not os.path.exists(out_root + 'validation_img/'):
		os.mkdir(out_root + 'validation_img/')
	if not os.path.exists(out_root + 'model/'):
		os.mkdir(out_root + 'model/')
	if not os.path.exists(out_root + 'validation_result/'):
		os.mkdir(out_root + 'validation_result/')
	if not os.path.exists(out_root + 'train_result/'):
		os.mkdir(out_root + 'train_result/')






	label_map = prodect_label(my_net.BATCH_SIZE)
	label_map_eval = prodect_label(my_net.EVAL_NUM)
	weight_label_train = prodect_instance_weight()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	out = open(('./log_2.txt'), 'wb')
	saver = tf.train.Saver()

	sess =  tf.InteractiveSession(config=config)

	tf.initialize_all_variables().run()
	print tf.is_variable_initialized(my_net.conv5_biases).eval()



	all_root = gd.generate_root('root_train.txt')
	all_z, all_x = gd.generate_pair_2(all_root, my_net.ALL_PAIR)


	all_root_eval = gd.generate_root('root_eval.txt')
	all_z_eval, all_x_eval = gd.generate_pair_2(all_root_eval, my_net.EVAL_NUM + 20)


	for k in range(my_net.EVAL_NUM):
		z_root = out_root + 'validation_img/' + '%d' % k + '_z.jpg'
		x_root = out_root + 'validation_img/' + '%d' % k + '_x.jpg'
		cv2.imwrite(z_root, all_z_eval[k])
		cv2.imwrite(x_root, all_x_eval[k])





	out_batch_z_eval, out_batch_x_eval = gd.generate_batch_2(my_net.EVAL_NUM, 0, all_z_eval, all_x_eval, my_net.EVAL_NUM + 20)
	out_batch_z_train, out_batch_x_train = gd.generate_batch_2(my_net.EVAL_NUM, 0, all_z, all_x, my_net.ALL_PAIR)

	tmp_1 = my_net.ALL_PAIR / my_net.BATCH_SIZE
	for epo_num in xrange(my_net.EPOC):
		for batch_num in xrange(my_net.ALL_PAIR / my_net.BATCH_SIZE):



			start_time = time.time()
			# out_batch_z, out_batch_x = gd.generate_batch(my_net.BATCH_SIZE, all_root)
			out_batch_z, out_batch_x = gd.generate_batch_2(my_net.BATCH_SIZE, batch_num, all_z, all_x, my_net.ALL_PAIR)

			# my_net.learning_rate = learning_rate[epo_num]
			_,loss_value = sess.run([my_net.optimizer, my_net.loss], feed_dict = {my_net.exempler_train_data:out_batch_z, 
				my_net.search_train_data:out_batch_x, my_net.train_label:label_map, my_net.weight_label:weight_label_train})

			duration = time.time() - start_time
			expected_time = ((my_net.EPOC - epo_num) * tmp_1 * duration + (tmp_1 - batch_num) * duration)/3600
			print ('epo_num: %d   epo_percent: %.2f   loss: %f    time: %.2f  expected_time: %.2fh'   %(epo_num, float(batch_num) / tmp_1, loss_value, duration, expected_time)) 
			out_string = '  ' + str(epo_num) + '  ' + str(batch_num)  + '  ' + str(loss_value) + '\n'
			out.write(out_string)



			if batch_num % 200 == 0:
				eval_resul = sess.run([my_net.eval_out], feed_dict = {my_net.exempler_eval_data:out_batch_z_eval, my_net.search_eval_data:out_batch_x_eval
					,my_net.weight_label:weight_label_train})
				eval_resul = np.asarray(eval_resul[0])
				for i in range(my_net.EVAL_NUM):
					tmp_eval_result = (eval_resul[i] - np.min(eval_resul[i]))/(np.max(eval_resul[i]) - np.min(eval_resul[i]))
					tmp_eval_result=cv2.resize(tmp_eval_result,(200, 200))
					tmp_eval_result = (tmp_eval_result * 255).astype(int)
					tmp_eval_result = np.asarray(tmp_eval_result)
					img_name = out_root + 'validation_result/'+ '%d' % epo_num + '_' + '%d' % batch_num + '_' + '%d' % i + '.jpg'
					cv2.imwrite(img_name, tmp_eval_result)


			# 	eval_resul = sess.run([my_net.eval_out], feed_dict = {my_net.exempler_eval_data:out_batch_z_train, my_net.search_eval_data:out_batch_x_train,
			# 		my_net.weight_label:weight_label_train})
			# 	eval_resul = np.asarray(eval_resul[0])
			# 	for i in range(my_net.BATCH_SIZE):
			# 		tmp_eval_result = (eval_resul[i] - np.min(eval_resul[i]))/(np.max(eval_resul[i]) - np.min(eval_resul[i]))
			# 		tmp_eval_result=cv2.resize(tmp_eval_result,(200, 200))
			# 		tmp_eval_result = (tmp_eval_result * 255).astype(int)
			# 		tmp_eval_result = np.asarray(tmp_eval_result)
			# 		img_name = out_root + 'train_result/' + '%d' % epo_num + '_' + '%d' % batch_num + '_' + '%d' % i + '.jpg'
			# 		cv2.imwrite(img_name, tmp_eval_result)





			if batch_num % 500 == 0:
				saver.save(sess, out_root + 'model/model.ckpt')

	sess.close()













