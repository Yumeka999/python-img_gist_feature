# -*- coding: utf-8 -*-

import os
import cv2
import sys
import time
import numpy as np

S_NOW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(S_NOW_DIR)

from utils_img import *

class GistUtils:
    def __init__(self, n_resize=128, n_w=5, ln_orientation=[8, 8, 8, 8], n_block_num=4, n_prefilt=4):
        self.n_resize = n_resize
        self.n_boundaryExtension = self.n_resize // 4
        self.n_w = n_w    
        self.ln_orientation = ln_orientation
        self.n_block_num = n_block_num  # MUST n_resize % n_block_num == 0
        self.n_prefilt = n_prefilt
             
        self.__create_gabor()
        self.__get_gfmat()
        
    def get_gist_vec(self, np_img_raw, mode="rgb"):
        # resize 
        np_img_resize, n_ret = img_resize(np_img_raw, self.n_resize)
        if n_ret != 0:
            print("image resize error")
            return None

        # convert gray or rgb
        np_gist = None
        if mode.lower() == "gray":
            np_img_gray, n_ret = img_2gray(np_img_resize)
            np_prefilt_img = self.__get_pre_filt(np_img_gray)
            np_gist = self.__gist_main(np_prefilt_img)

        elif mode.lower() == "rgb" or mode.lower() == "bgr":
            np_img_bgr, n_ret = img_2bgr(np_img_resize)
            
            np_img_b = np_img_bgr[:,:,0]
            np_img_g = np_img_bgr[:,:,1]
            np_img_r = np_img_bgr[:,:,2]

            np_gist_b = self.__get_pre_filt(np_img_b)
            np_gist_g = self.__get_pre_filt(np_img_g)
            np_gist_r = self.__get_pre_filt(np_img_r)

            np_gist_b = self.__gist_main(np_gist_b)
            np_gist_g = self.__gist_main(np_gist_g)
            np_gist_r = self.__gist_main(np_gist_r)


            np_gist = np.hstack([np_gist_b, np_gist_g, np_gist_r])
        else:
            print("input mode error")
        
        return np_gist
    
    def __get_pre_filt(self, np_img): 
        np_log_img = np.log(np_img + 1.0)
        np_pad_img = np.pad(np_log_img,((self.n_w,self.n_w), (self.n_w,self.n_w)), 'symmetric')

        np_gf = self.np_gf
        np_out = np_pad_img - np.real(np.fft.ifft2(np.fft.fft2(np_pad_img) * np_gf ))
        
        np_local = np.sqrt(np.abs(np.fft.ifft2(np.fft.fft2(np_out **2) * np_gf)))
        np_out = np_out / (0.2 + np_local)
        
        n_size = self.n_resize + 2 * self.n_w
        
        return np_out[self.n_w: n_size - self.n_w, self.n_w : n_size - self.n_w]
    
    def __gist_main(self, np_prefilt_img):
        
        n_b = self.n_boundaryExtension
        np_pad_img = np.pad(np_prefilt_img, ((n_b, n_b), (n_b, n_b)), 'symmetric')
        np_fft2_img = np.fft.fft2(np_pad_img)
        
    
        n_filter = self.np_gabor.shape[2]      
        n_size = self.np_gabor.shape[0]
        lf_gist = []
        for i in range(n_filter):
            np_res = np.abs(np.fft.ifft2( np_fft2_img * self.np_gabor[:,:,i] ))
            
            np_res = np_res[n_b: n_size - n_b, n_b : n_size - n_b]
            
            lf_filter_res = self.__down_sampling(np_res)      
            lf_gist = lf_gist + lf_filter_res
        
        np_gist = np.asarray(lf_gist)
        return np_gist[np.newaxis,:]
    
    def __create_gabor(self):
        n_gabor_size = self.n_resize + 2 * self.n_boundaryExtension
        ln_or = self.ln_orientation
        
        n_scales = len(ln_or)
        n_filters = sum(ln_or)
        
        np_param = np.zeros((n_filters, 4), dtype = np.float64)
        n_index = 0
        for i in range(n_scales):
            for j in range(0, ln_or[i]):
                np_param[n_index, 0] = 0.35
                np_param[n_index, 1] = 0.3 / (1.85**i)
                np_param[n_index, 2] = 16 *(ln_or[i]**2)/(32**2)
                np_param[n_index, 3] = np.pi/ln_or[i] * j
                
                n_index += 1
         
     
        np_linear = np.linspace(-n_gabor_size//2, n_gabor_size//2-1, n_gabor_size)
        np_fx, np_fy = np.meshgrid(np_linear, np_linear)
        np_res_A = np.fft.fftshift(np.sqrt(np_fx ** 2 + np_fy**2))
        np_res_B = np.fft.fftshift(np.angle(np_fx + 1j*np_fy))
        
        self.np_gabor = np.zeros((n_gabor_size, n_gabor_size, n_filters), dtype = np.float64)
        for i in range(n_filters):
            np_tr = np_res_B + np_param[i,3]
            np_A  = (np_tr < -np.pi) + 0.0
            np_B  = (np_tr > np.pi) + 0.0
            
            np_tr = np_tr + 2 *np.pi * np_A - 2*np.pi*np_B
            np_every_gabor = np.exp(-10 * np_param[i,0] * ((np_res_A / n_gabor_size /np_param[i,1] - 1) **2) - 2*np_param[i,2]*np.pi*(np_tr **2))
            
            self.np_gabor[:,:,i] = np_every_gabor
            
    def sav_gist_gabor_to_dir(self, s_gabor_folder = 'Gabor'):
        if self.np_gabor is None:
            print('gabor is not exists')

        if s_gabor_folder == None or s_gabor_folder == '':
            s_gabor_folder = 'Gabor'
            
        if not os.path.exists(s_gabor_folder) or not os.path.isdir(s_gabor_folder):
            os.makedirs(s_gabor_folder)
            
        n_row, n_col, n_num = self.np_gabor.shape
        for i in range(n_num):
            np_submat = self.np_gabor[:, :, i]
            np.savetxt('%s/Gabor%s' % (s_gabor_folder, str(i).zfill(2)), np_submat)               
        print('Write Success.')    
            
    
    def __get_gfmat(self):
        n_s1 = self.n_prefilt /np.sqrt(np.log(2))
        n_boundray = self.n_resize + 2 * self.n_w
         
        np_linear = np.linspace(-n_boundray//2, n_boundray//2-1, n_boundray)
        np_fx, np_fy = np.meshgrid(np_linear, np_linear)
        
#        np_gf = np.fft.fftshift(np.exp( -(np_fx **2 + np_fy **2)/(n_s1 ** 2)))
        self.np_gf = np.fft.fftshift(np.exp( -(np_fx **2 + np_fy **2)/(n_s1 ** 2)))
           
    def __down_sampling(self, np_img):
        np_index = np.linspace(0, self.n_resize, self.n_block_num + 1, dtype = np.int)
        ln_data = []
        for i in range(self.n_block_num):
            for j in range(self.n_block_num):
                np_zone = np_img[np_index[i]: np_index[i+1] , np_index[j]: np_index[j+1]]
                np_zone = np_zone.T.reshape(-1)
#                n_res = np.median(np_zone)
                n_res = np.max(np_zone)
                ln_data.append(n_res)
        return ln_data
    