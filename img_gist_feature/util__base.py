# -*- coding: utf-8 -*-

import os
import sys
import cv2
import shutil
import imghdr
import numpy as np
from PIL import Image


S_NOW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(S_NOW_DIR)



# Creat Directory
def recur_mkdir(s_path, run_logger=None, b_print=False):  
    if os.path.exists(s_path) and os.path.isdir(s_path):
        s_msg = "%s has been created" % s_path
        run_logger and run_logger.warning(s_msg)
        b_print and print(s_msg)
        return 1
    else:
        try:
            os.makedirs(s_path)
            
            s_msg = "%s create" % s_path
            run_logger and run_logger.info(s_msg)
            b_print and print(s_msg)
            return 0
        except Exception as e:
            s_msg = "%s" % str(e)
            run_logger and run_logger.error(s_msg)
            b_print and print(s_msg)      
            return -1


# Copy a file
def cp_file(s_in_url, s_out_url, run_logger=None, b_print=False):
    try: 
        shutil.copyfile(s_in_url, s_out_url) # Copy a old file to tmp dir 
        return 0
    except Exception as e:
        s_msg = 'Err: cant''t copy %s to %s, %s' % (s_in_url, s_out_url, str(e))
        run_logger and run_logger.error(s_msg)
        b_print and print(s_msg)
        return -1


# Copy a directory
def cp_dir(s_in_dir, s_out_dir, run_logger=None, b_print=False): 
    try:   
        n_ret = 0  
        if os.path.exists(s_out_dir) and os.path.isdir(s_out_dir):
            n_ret = rm_dir(s_out_dir)
        if os.path.exists(s_out_dir) and os.path.isfile(s_out_dir):
            n_ret = rm_file(s_out_dir)
        n_ret == 0 and shutil.copytree(s_in_dir, s_out_dir) # using copytree
        return n_ret
    except Exception as e:
        s_msg = 'Err: cant''t copy %s to %s, %s' % (s_in_dir, s_out_dir, str(e))
        run_logger and run_logger.error("%s" % s_msg)
        b_print and print(s_msg)
        return -1




# Move a file 
def mv_file(s_in_url, s_out_url, run_logger=None, b_print=False):
    try: 
        shutil.move(s_in_url, s_out_url) # using shutil.move
        return 0
    except Exception as e:
        s_msg = 'Err: cant''t copy %s to %s, %s' % (s_in_url, s_out_url, str(e))
        run_logger and run_logger.error(s_msg)
        b_print and print(s_msg)  
        return -1
    

# Delete a file
def rm_file(s_file, run_logger=None, b_print=False):
     try:
        if os.path.exists(s_file) and os.path.isfile(s_file): # judge is a file
            os.remove(s_file) 
            return 0
        else:
            s_msg = 'Err: not exists %s' % (s_file)
            run_logger and run_logger.warning(s_msg)
            b_print and print(s_msg)
            return 1
     except Exception as e:
         s_msg = 'Err: cant'' remove %s %s' % (s_file, str(e))
         run_logger and run_logger.error(s_msg)
         b_print and print(s_msg)
         return -1


# Delete a directory
def rm_dir(s_dir, run_logger=None, b_print=False):
     try:
        if os.path.exists(s_dir) and os.path.isdir(s_dir): # is a directory
            shutil.rmtree(s_dir) 
            return 0
        else:
            s_msg = 'Err: not exists %s' % (s_dir)
            run_logger and run_logger.error(s_msg)
            b_print and print(s_msg)
            return 1
     except Exception as e:
         s_msg = 'Err: cant'' remove %s' % (s_dir, str(e))
         run_logger and run_logger.error(s_msg)
         b_print and print(s_msg)
         return -1

# Get right string
def get_usable_str(s_in):
    s_tmp = s_in  
    s_unvalid = '<>,\/|,:.,''",*,?\t\r\n'
    for ch in s_unvalid:
        s_tmp = s_tmp.replace(ch,'')
    s_tmp = s_tmp.replace(u'\u3000','')
    s_tmp = s_tmp.replace('🔥','')
    return s_tmp

# Get all cos sim
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
