import cv2
import numpy as np
# import generate_data as gd
# import tensorflow as tf


# aa = []
# for k in range(64):
# 	root_x = './eval_img_0.0001/' + '%d' %k + '_x.jpg'
# 	img = cv2.imread(root_x)
# 	aa.append(img)
# 	print aa[0]
# 	break
# 	img = 255 - img
# 	cv2.imwrite(root_x, img)
# 	root_z = './eval_img_0.0001/' + '%d' %k + '_z.jpg'
# 	img = cv2.imread(root_z)
# 	img = 255 - img
# 	cv2.imwrite(root_z, img)



# all_root_eval = gd.generate_root('root_eval.txt')
# all_z_eval, all_x_eval = gd.generate_pair_2(all_root_eval, 5)

# print all_z_eval[0]


file = open('lr.txt')

for i in file:
	aa = i.split('\t')

# print float(aa[0])
print float(aa[49])