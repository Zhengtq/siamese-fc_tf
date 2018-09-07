#!/usr/bin/python
#-*-coding:utf-8 -*-

import numpy as np
import time
import math
import functions as fun
import cv2

SEED = 66478 
BATCH_SIZE = 16
EXEMPLAR_IMAGE_SIZE = 127
SEARCH_IMAGE_SIZE = 255
NUM_CHANNELS = 3
epsilon = 1e-3
context_amount = 0.5
instance_size = 255
responseUp = 16
scoreSize = 17
scaleLR = 0.59

#target_position是目标的中心位置
def tracker(img_files, target_position, target_size, file_name, sess, my_net, bd_root):


	out_file = open((bd_root + file_name + '.txt'), 'wb')

	frame_nums = len(img_files) + 1

	[hc_z, wc_z] = target_size + context_amount * np.sum(target_size)

	#目标最终的长和宽（矩形）
	s_z = math.sqrt(wc_z * hc_z)
	scale_z = EXEMPLAR_IMAGE_SIZE / s_z
	im = img_files[0]

	z_corp= fun.get_subwindow_tracking(im, target_position, [EXEMPLAR_IMAGE_SIZE, EXEMPLAR_IMAGE_SIZE], np.array([round(s_z), round(s_z)]))


	z_corp = np.expand_dims(z_corp, axis=0)

	d_search = (instance_size - EXEMPLAR_IMAGE_SIZE)/2
	pad = d_search/scale_z
	s_x = s_z + 2*pad
	min_s_x = 0.2*s_x
	max_s_x = 5*s_x
	tmp_x_window = np.hanning(responseUp * scoreSize)
	tmp_y_window = np.hanning(responseUp * scoreSize)
	tmp_y_window.shape = (responseUp * scoreSize, 1)
	window = tmp_x_window * tmp_y_window
	window = window/np.sum(window)
	scales = np.array([0.9639, 1.0, 1.0375])

	z_feature = sess.run(my_net.out_feature, feed_dict = {my_net.exempler_test_data:z_corp})


	for i in xrange(len(img_files)):

		print i
		if i > 0:
			im = img_files[i]
			scaled_instance = scales * s_x
			scaled_target = [target_size[0] * scales]
			scaled_target.append(target_size[1] * scales)

			x_crops = fun.make_scale_pyramid(im, target_position, scaled_instance, instance_size)


			response_map = []
			response_map.append(sess.run(my_net.test_prediction, feed_dict = {my_net.exempler_feature_outside:z_feature, my_net.search_test_data:x_crops[0]}))
			response_map.append(sess.run(my_net.test_prediction, feed_dict = {my_net.exempler_feature_outside:z_feature, my_net.search_test_data:x_crops[1]}))
			response_map.append(sess.run(my_net.test_prediction, feed_dict = {my_net.exempler_feature_outside:z_feature, my_net.search_test_data:x_crops[2]}))




			new_target_position,new_scale = fun.tracker_eval(s_x, target_position, window, response_map)
			target_position = new_target_position
			s_x = max(min_s_x, min(max_s_x, (1-scaleLR)*s_x + scaleLR*scaled_instance[new_scale]))
			target_size = (1-scaleLR)*target_size + scaleLR*np.array([scaled_target[0][new_scale], scaled_target[1][new_scale]])



		final_box = [target_position[1] -  target_size[1]/2, target_position[0] -  target_size[0]/2, target_size[1], target_size[0]]

		out_file.write(str(int(final_box[0]))  + ' ')
		out_file.write(str(int(final_box[1]))  + ' ')
		out_file.write(str(int(final_box[2]))  + ' ')
		out_file.write(str(int(final_box[3]))  + ' ')
		out_file.write(' \n')
	out_file.close()




