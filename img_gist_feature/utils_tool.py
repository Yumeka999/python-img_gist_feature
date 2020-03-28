# -*- coding: utf-8 -*-

import os
import cv2
import time
import imghdr
import shutil
import numpy as np

# @Create dir 
# @input:s_path dir to save
# @output:
def recur_mkdir(s_path, run_logger=None):  
    if os.path.exists(s_path) and os.path.isdir(s_path):
        run_logger and run_logger.info("%s has been created" % s_path)
    else:
        os.makedirs(s_path)
        run_logger and run_logger.info("%s create" % s_path)

# Get right string
def get_usable_str(s_in):
    s_tmp = s_in  
    s_unvalid = '<>,\/|,:.,''",*,?\t\r\n'
    for ch in s_unvalid:
        s_tmp = s_tmp.replace(ch,'')
    s_tmp = s_tmp.replace(u'\u3000','')
    return s_tmp

# 删除一个文件
def rm_file(s_file, run_logger=None):
     try:
        if os.path.exists(s_file) and os.path.isfile(s_file): # 旧文件存在则删除
            os.remove(s_file) 
        else:
            run_logger and run_logger.warning('Err: not exists %s' % (s_file, str(e)))
     except Exception as e:
         run_logger and run_logger.error('Err: cant'' remove %s' % (s_file, str(e)))
         return -1
     return 0

# 删除一个文件夹
def rm_dir(s_dir, run_logger=None):
     try:
        if os.path.exists(s_dir) and os.path.isdir(s_dir): # 旧文件存在则删除
            shutil.rmtree(s_dir) 
        else:
            run_logger and run_logger.error('Err: not exists %s' % (s_file, str(e)))
     except Exception as e:
         run_logger and run_logger.error('Err: cant'' remove %s' % (s_file, str(e)))
         return -1
     return 0

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

def get_cos_sim(np_A, np_B):
    return np.inner(np_A, np_B)/(np.linalg.norm(np_A) * np.linalg.norm(np_B))


# 复制一个文件
def cp_file(s_in_url, s_out_url, run_logger=None):
    try: 
        shutil.copyfile(s_in_url, s_out_url) # 旧文件复制到临时文件夹中 
    except Exception as e:
        run_logger and run_logger.error('Err: cant''t copy %s to %s, %s' % (s_in_url, s_out_url, str(e)))
        return -1
    return 0

# 移动一个文件
def mv_file(s_in_url, s_out_url, run_logger=None):
    try: 
        shutil.move(s_in_url, s_out_url) # 旧文件复制到临时文件夹中 
    except Exception as e:
        run_logger and run_logger.error('Err: cant''t copy %s to %s, %s' % (s_in_url, s_out_url, str(e)))
        return -1
    return 0