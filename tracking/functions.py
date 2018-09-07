#!/usr/bin/python
#-*-coding:utf-8 -*-

import cv2
import numpy as np
import math
SEARCH_IMAGE_SIZE = 255
scoreSize = 17
def get_subwindow_tracking(im, target_position, model_sz, original_sz):
	
	
    im_sz = [im.shape[0], im.shape[1]]

    half_size = (original_sz + 1)/2



    context_xmin = round(target_position[1] - half_size[1])
    context_xmax = context_xmin + original_sz[1] - 1
    context_ymin = round(target_position[0] - half_size[0])
    context_ymax = context_ymin + original_sz[0] - 1


    left_pad = max(0, 1- context_xmin)
    top_pad = max(0, 1- context_ymin)
    right_pad = max(0, context_xmax - im_sz[1])
    bottom_pad = max(0, context_ymax - im_sz[0])



    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)



    if top_pad or left_pad or bottom_pad or right_pad:
        top_pad = int(top_pad)
        left_pad = int(left_pad)
        bottom_pad = int(bottom_pad)
        right_pad = int(right_pad)
        R = np.lib.pad(im[:,:,0], ((top_pad,bottom_pad), (left_pad, right_pad)),  'mean')
        G = np.lib.pad(im[:,:,1], ((top_pad,bottom_pad), (left_pad, right_pad)),  'mean')
        B = np.lib.pad(im[:,:,2], ((top_pad,bottom_pad), (left_pad, right_pad)),  'mean')

        im = np.zeros((R.shape[0],R.shape[1], 3))
        im[:,:,0] = R
        im[:,:,1] = G
        im[:,:,2] = B



    im_patch_original = im[context_ymin : context_ymax, context_xmin : context_xmax, :]
    im_patch = cv2.resize(im_patch_original, (model_sz[0], model_sz[0]))


    return im_patch




def make_scale_pyramid(im, target_position, in_side_scaled, out_side):
    in_side_scaled = [round(tmp) for tmp in in_side_scaled]
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side
    search_side = int(beta * max_target_side)
    search_region = get_subwindow_tracking(im, target_position, [search_side,search_side], np.array([max_target_side,max_target_side]))


    pyramid = []
    for s in range(3):
        target_side = round(beta * in_side_scaled[s])
        tmp_input = [(search_side + 1)/2, (search_side + 1)/2]
        tmp_x_corp = get_subwindow_tracking(search_region, tmp_input, [SEARCH_IMAGE_SIZE,SEARCH_IMAGE_SIZE], np.array([target_side,target_side]))
        tmp_x_corp = np.expand_dims(tmp_x_corp, axis=0)
        pyramid.append(tmp_x_corp)


    return pyramid



def tracker_eval(s_x, target_position, window, response_map):


    current_scale_ID = 1
    best_scale = 1
    best_peak = -float('Inf')
    response_map_up = []
    response_up = 16
    scale_penalty = 0.9745



    wInfluence = 0.176


    score_size = 17
    total_stride = 8
    instance_size = 255


    for s in range(3):
        this_response=cv2.resize(response_map[s],(response_up * score_size, response_up * score_size),interpolation=cv2.INTER_CUBIC)

        response_map_up.append(this_response)

        if s != 1:
            this_response = this_response * scale_penalty

        this_peak = np.amax(this_response)
        if this_peak > best_peak:
            best_peak = this_peak
            best_scale = s


    response_map = response_map_up[best_scale]
    response_map = response_map - np.amin(response_map)
    response_map = response_map / np.sum(response_map)




    response_map = (1-wInfluence)*response_map + wInfluence*window
    r_max,c_max = np.unravel_index(response_map.argmax(), response_map.shape)





    p_corr = np.array([r_max, c_max])
    disp_instance_final = p_corr - math.ceil(score_size*response_up/2)



    disp_instance_input = disp_instance_final * total_stride / response_up
    disp_instance_frame = disp_instance_input * s_x / instance_size
    new_target_position = target_position + disp_instance_frame

    return new_target_position,  best_scale












