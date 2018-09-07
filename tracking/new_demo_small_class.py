#!/usr/bin/python
#-*-coding:utf-8 -*-

import numpy as np
import os
import re
import random
import cv2
import tracking_class 
import tensorflow as tf
import sys
sys.path.append('../')
import my_net as mn
import os

def search(path):

    path = os.path.expanduser(path)
    dir_all = []
    for f in os.listdir(path):
    	root = path + f.strip()
    	dir_all.append(root)
    return dir_all



def demo():


	all_root = 'test/'
	model_root = all_root + 'model/'
	bd_root = all_root + 'myBD/'

	if not os.path.exists(all_root):
		os.mkdir(all_root)
	if not os.path.exists(model_root):
		os.mkdir(model_root)
	if not os.path.exists(bd_root):
		os.mkdir(bd_root)



	params = {'seed' : 66478, 'batch_size' : 32, 'exemplar_image_size' : 127, 
	'search_image_size' : 255, 'img_channels' : 3, 'batch_norm_eps' : 1e-3, 
	'all_pair' : 50000, 'epoc' : 50, 'eval_num' : 64}
	my_net = mn.My_net(params)

	saver = tf.train.Saver()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	sess = tf.Session(config=config) 
	saver.restore(sess, model_root + 'model.ckpt')
	root = '~/sequence/'
	dir_all = search(root)

	for k in xrange(len(dir_all)):

		sequence_name = dir_all[k].split('/')[-1]

		# if sequence_name != 'Doll':
		# 	continue

		print sequence_name
		img_dir = dir_all[k] + '/img/'

		path = os.path.expanduser(img_dir)
		frame_nums = 0
		for f in os.listdir(path):
	   		frame_nums = frame_nums + 1

		img_data = []
		for i in range(frame_nums):
			print i
	   		tmp_img_root = img_dir + '%04d' % (i + 1) + '.jpg'
	   		tmp_img = cv2.imread(tmp_img_root)
	   		tmp_img = np.array(tmp_img) 
			tmp_img = tmp_img/255.0
			tmp_img = tmp_img.astype(np.float32)
			img_data.append(tmp_img)


		groundtruth_root = dir_all[k] + '/groundtruth_rect.txt'

		gt_text = open(groundtruth_root, 'r')
		line = gt_text.readline()
		gt_text.close()

		if re.search(',', line):
			line = line.split(',')
		else:
			line = line.split()

		x = int(line[0])
		y = int(line[1])
		w = int(line[2])
		h = int(line[3])

		pos = np.array([y + h/2, x + w/2])
		target_sz = np.array([h, w])

		tracking_class.tracker(img_data, pos, target_sz, sequence_name, sess, my_net, bd_root)

	sess.close()



demo()
