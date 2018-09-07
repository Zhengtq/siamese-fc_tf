import numpy as np
import os
import re
import random
import cv2

path_a = '/data2/zhengtq/data/imagenet_tracking/a/'
path_b = '/data2/zhengtq/data/imagenet_tracking/b/'
path_c = '/data2/zhengtq/data/imagenet_tracking/c/'
path_d = '/data2/zhengtq/data/imagenet_tracking/d/'
path_e = '/data2/zhengtq/data/imagenet_tracking/e/'


def search(path):
    path = os.path.expanduser(path)
    for f in os.listdir(path):
    	root = path + f.strip() + '/'
    	dir_all.append(root)




dir_all = []
def exhibit_all():

	search(path_a)
	search(path_b)
	search(path_c)
	search(path_d)
	search(path_e)
	return dir_all




def train_batch():
	exhibit_all()

	out = open(('./root.txt'), 'wb')

	for i in range(len(dir_all)):
		print i
		out.write(dir_all[i] + '  ')
		for root, dis, files in os.walk(dir_all[i]):  

			frames = 0
			objects = 0
			for file1 in files:   
				tmp = file1.split('.')
				frames = max(int(tmp[0]), frames)
				objects = max(int(tmp[1]), objects) 

		out.write(str(frames) + '  ')
		out.write(str(objects))
		out.write('\n')
	out.close()

#train_batch()



def generate_root(root):
	infile = open(root, 'r')
	all_root = []
	for i in infile:
		all_root.append(i.split())
	infile.close()
	return all_root





def generate_pair(all_root):

	video_num = random.randint(0, len(all_root) - 1)
	out_root = []
	while(1):
		z_frame_num = random.randint(0, int(all_root[video_num][1]))
		object_num = random.randint(0, int(all_root[video_num][2]))
		z_path = all_root[video_num][0] + '%06d' % z_frame_num + '.' + '%02d' % object_num + '.crop.z.jpg'


		if os.path.exists(z_path):
			x_frame_num = random.randint(max(0, z_frame_num - 100), min(z_frame_num + 100, int(all_root[video_num][1])))

			# x_frame_num_1 = range(0, max(1, z_frame_num - 100))
			# x_frame_num_2 = range(min(z_frame_num + 100, int(all_root[video_num][1]) - 1), int(all_root[video_num][1]))
			# x_frame_num_1.extend(x_frame_num_2)
			# x_frame_num = random.choice(x_frame_num_1)

			x_path = all_root[video_num][0] + '%06d' % x_frame_num + '.' + '%02d' % object_num + '.crop.x.jpg'
			if os.path.exists(x_path) and z_frame_num != x_frame_num:

				image_z = cv2.imread(z_path)
				image_x = cv2.imread(x_path)
				

				out_root.append(image_z)
				out_root.append(image_x)
				
				break

	return out_root


def generate_batch(batch_size, all_root):


	out_batch_z = []
	out_batch_x = []
	
	for i in range(batch_size):
		tmp_pair = generate_pair(all_root)
		out_batch_z.append(tmp_pair[0])
		out_batch_x.append(tmp_pair[1])


	out_batch_z = np.array(out_batch_z)
	out_batch_x = np.array(out_batch_x) 
	

	out_batch_z = out_batch_z/255.0
	out_batch_x = out_batch_x/255.0
	

	out_batch_z = out_batch_z.astype(np.float32)
	out_batch_x = out_batch_x.astype(np.float32)
	

	return out_batch_z, out_batch_x


#generate_batch(8)



def generate_pair_2(all_root, ALL_PAIR):

	out_root_z = []
	out_root_x = []
	count = 0
	while(count < ALL_PAIR):
		video_num = random.randint(0, len(all_root) - 1)

		if int(all_root[video_num][2]) != 0:
			continue


		z_frame_num = random.randint(0, int(all_root[video_num][1]))
		# object_num = random.randint(0, int(all_root[video_num][2]))
		object_num = 0
		z_path = all_root[video_num][0] + '%06d' % z_frame_num + '.' + '%02d' % object_num + '.crop.z.jpg'


		if os.path.exists(z_path):
			x_frame_num = random.randint(max(0, z_frame_num - 100), min(z_frame_num + 100, int(all_root[video_num][1])))

			x_path = all_root[video_num][0] + '%06d' % x_frame_num + '.' + '%02d' % object_num + '.crop.x.jpg'
			if os.path.exists(x_path) and z_frame_num != x_frame_num:

				image_z = cv2.imread(z_path)
				image_x = cv2.imread(x_path)

				if (image_z is None) or (image_x is None):
					continue

				out_root_z.append(image_z)
				out_root_x.append(image_x)

				count = count + 1
				if count % 100 == 0:
					print count
				

	return out_root_z, out_root_x




def generate_batch_2(batch_size, the_batch,all_z, all_x, ALL_PAIR):


	
	out_batch_z = []
	out_batch_x = []

	for i in range(batch_size):

		img_num = the_batch * batch_size + i

		if img_num > ALL_PAIR - 1:
			img_num = img_num - ALL_PAIR

		# image_z = cv2.imread(all_z[img_num])
		# image_x = cv2.imread(all_x[img_num])

		out_batch_z.append(all_z[img_num])
		out_batch_x.append(all_x[img_num])


	out_batch_z = np.array(out_batch_z)
	out_batch_x = np.array(out_batch_x) 
	


	out_batch_z = out_batch_z/255.0
	out_batch_x = out_batch_x/255.0
	
	out_batch_z = out_batch_z.astype(np.float32)
	out_batch_x = out_batch_x.astype(np.float32)
	

	return out_batch_z, out_batch_x


