# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 07:41:30 2020

@author: Abubakar
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
def convolution(input_image, mask,stride = 1,bp = False):
    
    i_row, i_col,i_ch = input_image.shape
    k_row, k_col,k_ch = mask.shape
    output = np.zeros((input_image.shape[0],input_image.shape[1]))
    pad_height = ((i_row*stride)-i_row+k_row - stride) // 2
    pad_width = ((i_col*stride)-i_col+k_col - stride) // 2
 
    padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width),i_ch))
    mask_without_ch = np.zeros((k_row,k_col))
    for ch in range(i_ch):
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width,ch] = input_image[:,:,ch]
    for ch in range(i_ch):
        for row in range(i_row):
            for col in range(i_col):
                mask_without_ch = mask[:,:,ch]
                if(bp):
                    mask_without_ch = np.flip(mask_without_ch)
                output[row, col] = output[row, col] + np.sum(mask_without_ch * padded_image[row:row + k_row, col:col + k_col,ch])
    return output
def convolve(input_image, size = 2,stride = 2):
    
        i_row, i_col,i_ch = input_image.shape
        k_row, k_col = size,size
        new_row = ((i_row - size)//stride) + 1
        new_col = ((i_col - size)//stride) + 1
        output = np.zeros((new_row,new_col,i_ch))
        #pad_height = ((i_row*stride)-i_row+k_row - stride) // 2
        #pad_width = ((i_col*stride)-i_col+k_col - stride) // 2
        #padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width),i_ch))
        #for ch in range(i_ch):
        #    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width,ch] = input_image[:,:,ch]
        for ch in range(i_ch):
            for row in range(new_row):
                for col in range(new_row):
                    output[row,col,ch] = np.max(input_image[(row*stride):(row*stride) + k_row, (col * stride):(col * stride) + k_col,ch])
        return output
A = np.arange(64).reshape((4,4,4))
A1 = convolve(A, size = 2,stride = 2)
#A = np.flip(A)
#a = A.reshape(A.shape[0]*A.shape[1]*A.shape[2], 1)
#a1 =A.flatten()
#A = np.flip(A, 1)
# =============================================================================
# input_image = cv.imread('book_gray.png')
# averaging_filter = 1/9*np.ones((3,3,3))
# vertical_edge_detection = np.array([[[1, 0, -1], [1, 0, -1], [1, 0, -1]],[[1, 0, -1], [1, 0, -1], [1, 0, -1]],[[1, 0, -1], [1, 0, -1], [1, 0, -1]]])
# horizontal_edge_detection = np.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])
# #vertical_edge_detection = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]],[[1, 0, -1], [1, 0, -1], [1, 0, -1]],[[1, 0, -1], [1, 0, -1], [1, 0, -1]])
# #horizontal_edge_detection = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],[[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
# 
# no_fil = 3
# filters = np.zeros((3,3,3,3))
# filters[0,:,:,:] = averaging_filter
# filters[1,:,:,:] = vertical_edge_detection
# filters[2,:,:,:] = horizontal_edge_detection
# output = np.zeros((input_image.shape[0],input_image.shape[1],no_fil))
# output[:,:,0] = convolution(input_image, filters[0,:,:,:])
# output[:,:,1] = convolution(input_image, filters[1,:,:,:])
# output[:,:,2] = convolution(input_image, filters[2,:,:,:])
# plt.imshow(input_image)
# plt.figure()
# plt.imshow(output[:,:,0])
# plt.figure()
# plt.imshow(output[:,:,1])
# plt.figure()
# plt.imshow(output[:,:,2])
# plt.imsave("result_avg_filter2.png",output[:,:,0])
# plt.imsave("result_ver_edge_detection2.png",output[:,:,1])
# plt.imsave("result_hor_edge_detection2.png",output[:,:,2]) 
# =============================================================================

# =============================================================================
# result_avg_filter = convolution(input_image, averaging_filter)
# result_ver_edge_detection = convolution(input_image, vertical_edge_detection)
# result_hor_edge_detection = convolution(input_image, horizontal_edge_detection)
# plt.imshow(input_image)
# plt.figure()
# plt.imshow(result_avg_filter)
# plt.figure()
# plt.imshow(result_ver_edge_detection)
# plt.figure()
# plt.imshow(result_hor_edge_detection)
# plt.imsave("result_avg_filter2.png",result_avg_filter)
# plt.imsave("result_ver_edge_detection2.png",result_ver_edge_detection)
# plt.imsave("result_hor_edge_detection2.png",result_hor_edge_detection)
# =============================================================================
