# -*- coding: utf-8 -*-
import numpy as np
import cv2
# import minpy.numpy as np
import os
#import matplotlib.pyplot as plt
import time
#import tensorflow as tf

class GistUtils:
    def __init__(self, o_param = None):
        if o_param is None:
            self.n_boundaryExtension = 32
            self.n_w = 5
            self.n_resize = 128
            self.ln_orientation = [8, 8, 8, 8]
            self.n_block_num = 8
            self.n_prefilt = 4
        else:
            self.n_boundaryExtension = o_param['n_boundaryExtension']
            self.n_w = o_param['n_w']
            self.n_resize = o_param['n_resize']
            self.ln_orientation = o_param['ln_orientation']
            self.n_block_num = o_param['n_block_num']
            self.n_prefilt = o_param['n_prefilt']
            
        self.create_gabor()
        self.get_gfmat()
        
    def get_gist_vec(self, np_img):
        np_prefilt_img = self.get_pre_filt(np_img)
        np_gist = self.gist_main(np_prefilt_img)
        return np_gist
    
    def get_pre_filt(self, np_img): 
        np_log_img = np.log(np_img + 1.0)
        np_pad_img = np.pad(np_log_img,((self.n_w,self.n_w), (self.n_w,self.n_w)), 'symmetric')

        np_gf = self.np_gf
        np_out = np_pad_img - np.real(np.fft.ifft2(np.fft.fft2(np_pad_img) * np_gf ))
        
        np_local = np.sqrt(np.abs(np.fft.ifft2(np.fft.fft2(np_out **2) * np_gf)))
        np_out = np_out / (0.2 + np_local)
        
        n_size = self.n_resize + 2 * self.n_w
        
        return np_out[self.n_w: n_size - self.n_w, self.n_w : n_size - self.n_w]
    
    def gist_main(self, np_prefilt_img):
        
        n_b = self.n_boundaryExtension
        np_pad_img = np.pad(np_prefilt_img, ((n_b, n_b), (n_b, n_b)), 'symmetric')
        np_fft2_img = np.fft.fft2(np_pad_img)
        
    
        n_filter = self.np_gabor.shape[2]      
        n_size = self.np_gabor.shape[0]
        lf_gist = []
        for i in range(n_filter):
            np_res = np.abs(np.fft.ifft2( np_fft2_img * self.np_gabor[:,:,i] ))
            
            np_res = np_res[n_b: n_size - n_b, n_b : n_size - n_b]
            
            lf_filter_res = self.down_sampling(np_res)      
            lf_gist = lf_gist + lf_filter_res
        
        np_gist = np.asarray(lf_gist)
        return np_gist[np.newaxis,:]
    
    def create_gabor(self):
        n_gabor_size = self.n_resize + 2 * self.n_boundaryExtension
        ln_or = self.ln_orientation
        
        n_scales = len(ln_or)
        n_filters = sum(ln_or)
        
        np_param = np.zeros((n_filters,4), dtype = np.float64)
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
            os.mkdir(s_gabor_folder)
            
        n_row, n_col, n_num = self.np_gabor.shape
        for i in range(n_num):
            np_submat = self.np_gabor[:, :, i]
            np.savetxt('%s/Gabor%s' % (s_gabor_folder, str(i).zfill(2)), np_submat)               
        print('Write Success.')    
            
    
    def get_gfmat(self):
        n_s1 = self.n_prefilt /np.sqrt(np.log(2))
        n_boundray = self.n_resize + 2 * self.n_w
         
        np_linear = np.linspace(-n_boundray//2, n_boundray//2-1, n_boundray)
        np_fx, np_fy = np.meshgrid(np_linear, np_linear)
        
#        np_gf = np.fft.fftshift(np.exp( -(np_fx **2 + np_fy **2)/(n_s1 ** 2)))
        self.np_gf = np.fft.fftshift(np.exp( -(np_fx **2 + np_fy **2)/(n_s1 ** 2)))
           
    def down_sampling(self, np_img):
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
    
def preproc_img(s_img_url):
    np_raw_img = cv2.imread(s_img_url, cv2.IMREAD_UNCHANGED)
    n_row = np_raw_img.shape[0]
    n_col = np_raw_img.shape[1]
    
    if n_row!=128 or n_col!=128:
        np_resize_img = cv2.resize(np_raw_img, (128,128), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
    else:
        np_resize_img = np_raw_img
    # Raw Image is BGR imge, so convert rgb to gray 
    if len(np_resize_img.shape) == 3 and np_resize_img.shape[2] == 3:
        np_gray_img = cv2.cvtColor(np_resize_img, cv2.COLOR_BGR2GRAY)
    
    # Raw Image is BGRA imge, there are different situation to solve
    elif len(np_resize_img.shape) == 3 and np_resize_img.shape[2] == 4: 
        n_sence = 3
        np_gray_img_choose = np.zeros([np_resize_img.shape[0], np_resize_img.shape[1], n_sence], dtype= np.uint8)
        
        np_gray_img_choose[:,:,0] = 255 - np_resize_img[:,:,3]   
        np_gray_img_choose[:,:,1] = cv2.cvtColor(np_resize_img, cv2.COLOR_BGRA2GRAY)      
        np_gray_img_choose[:,:,2] = cv2.cvtColor(np_resize_img[:,:,0:3], cv2.COLOR_BGR2GRAY)  
       
        # Get nonzero element of every resize gray 
        ln_sence_non0_num = []
        for i in range(n_sence):
            ln_sence_non0_num.append(len(np_gray_img_choose[:,:,i].nonzero()[0]))
        
        # Which image has most nonzero element
        if len(set(ln_sence_non0_num)) > 1:
            n_max_index = ln_sence_non0_num.index(max(ln_sence_non0_num))    
            np_gray_img = np_gray_img_choose[:,:,n_max_index]
        else:  
            # Which image has most different element  
            ln_diff_pix_num = [] 
            for i in range(n_sence):
                ln_diff_pix_num.append(len(np.unique(np_gray_img_choose[:,:,i])))
            n_max_index = ln_diff_pix_num.index(max(ln_diff_pix_num))    
            np_gray_img = np_gray_img_choose[:,:,n_max_index]
        
    # Raw Image is gray image
    elif len(np_resize_img.shape) == 3 and np_resize_img.shape[2] == 1:
        np_gray_img = np_resize_img[:,:,0]
    elif len(np_resize_img.shape) == 2:
        np_gray_img = np_resize_img
    
    return np_gray_img

def get_cos_sim(np_A, np_B):
    return np.inner(np_A, np_B)/(np.linalg.norm(np_A) * np.linalg.norm(np_B))



#@numba.autojit
def get_all_cos_sim(np_A ,np_B, np_B_L2 = None):
    n_num = np_B.shape[0]
    
    t1 = time.clock()  
    np_A_L2 = np.linalg.norm(np_A)
    np_A_L2 = np.tile(np_A_L2, (1, n_num))   
    t2 = time.clock()
    print('A L2 time',t2 - t1)
    
    if np_B_L2 is None:
        t1 = time.clock()  
        np_B_L2 = np.linalg.norm(np_B, axis = 1)
        t2 = time.clock()
        print('B L2 time',t2 - t1)
    
    
    t1 = time.clock()
    np_inner = np_A.dot(np_B.T)
    

    np_cos_sim = np_inner/(np_A_L2 * np_B_L2)
    np_cos_sim = 0.5 + 0.5 * np_cos_sim
    t2 = time.clock()
    print('cos time', t2 -t1)
    
    return np_cos_sim
    
def np_l2norm(np_x):
    if len(np_x.shape) > 2:
        return -1
    elif len(np_x.shape) == 1:
        np_x = np_x[:, np.newaxis]
        np_x = np_x.T
        
    # the feature number of input
    n_feat_num = np_x.shape[1]    
    
    np_x_L2 = np.linalg.norm(np_x, axis = 1)
    np_x_L2 = np_x_L2[:, np.newaxis]
    np_x_L2 = np.tile(np_x_L2, (1,n_feat_num))
    np_x_L2_1 = 1.0/np_x_L2
    
#    print(np_x_L2.shape)
    np_x_L2Norm = np_x * np_x_L2_1
    
    # if value is nan and set 0.0
    np_where_are_nans = np.isnan(np_x_L2Norm)
    np_x_L2Norm[np_where_are_nans] = 0.0
    
    return np_x_L2Norm

if __name__ == '__main__':
    gist_helper = GistUtils()
    gist_helper.sav_gist_gabor_to_dir('')
#    print(np_resize_img)
#    plt.imshow(np_resize_img)
    
    
    s_img_url_A = './data/Photo_899/000002_1036871_pingmianjihe.jpg'
    np_resize_img_A = preproc_img(s_img_url_A)
    np_gist_A = gist_helper.get_gist_vec(np_resize_img_A)
    
    print(np_gist_A)
#    
#    
#    s_img_url_B = './Photo_899/000005_1048896.jpg'
#    np_resize_img_B = PreprcessImg(s_img_url_B)
#    np_gist_B = gist_helper.GetGistVec(np_resize_img_B)
#    
#    print(GetCosSim(np_gist_A, np_gist_B))
#    np_gist = np_gist_B[np.newaxis, :]
       
#    f_cost_time = 0.0
#    for i in range(20):
#        t1 = time.clock()
#        np_gist = gist_helper.GetGistVec(np_resize_img)
#        t2 = time.clock()
#        print('get vec time',t2- t1)
#        
#        f_cost_time += t2 -t1
#    print('cost time:', f_cost_time/20)
    
#    B = np.random.rand(100000,2048)
    
#    A = np.tile(np_gist, (1,1))
#    B = np.tile(np_gist, (100000 ,1))
#    np_B_L2 = np.linalg.norm(B, axis = 1)
#    
#    target_index = [-5,-4,-3,-2,-1]
#    
#
#    f_cost_time = 0.0
#    for i in range(20):
#        t1 = time.clock()        
#        np_cos_sim = GetAllCosSim(A ,B, np_B_L2)
#        np_cos_sim_sorted = np.argsort(np_cos_sim).reshape(-1) 
#        topK_index = np.take(np_cos_sim_sorted, target_index)
#        t2 = time.clock()
##        print(np_cos_sim)
#        print('mat Time:',t2-t1)
#        print()
#        f_cost_time += t2 - t1
#          
#    print('Numpy mat cost time:', f_cost_time/20)
#    print()
        
            
#    f_cost_time = 0.0
#    for i in range(20):   
#        t1 = time.clock()  
#        
#        np_cos_sim = np.zeros((100000,1))
#        for j in range(100000):
#            np_cos_sim[j] = GetCosSim(A, B[j,:])
#            
#        np_cos_sim_sorted = np.argsort(np_cos_sim).reshape(-1) 
#        topK_index = np.take(np_cos_sim_sorted, target_index)
#        t2 = time.clock()
#        print('mat Time:',t2-t1)
#  
#        f_cost_time += t2 - t1
#          
#    print('one step cos cost time:', f_cost_time/20)

        
        
        