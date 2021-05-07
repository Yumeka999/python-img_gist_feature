# -*- coding: utf-8 -*-
import os
import time
import numpy as np
'''
Get all cos sim
'''
def get_all_cos_sim(np_A ,np_B, np_B_L2 = None, run_logger=None, b_print=False):
    n_num = np_B.shape[0] 
    np_A_L2 = np.linalg.norm(np_A)
    np_A_L2 = np.tile(np_A_L2, (1, n_num))   
    if np_B_L2 is None: np_B_L2 = np.linalg.norm(np_B, axis = 1)
    np_inner = np_A.dot(np_B.T)    
    np_cos_sim = 0.5 + 0.5 * np_inner/(np_A_L2 * np_B_L2)
    return np_cos_sim
'''
get l2 norm of a vector
'''  
def np_l2norm(np_x, run_log=None, b_print=False):
    if len(np_x.shape) > 2: 
        s_msg = "input vector shape > 2"
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return -1
    elif len(np_x.shape) == 1: 
        np_x = np_x[:, np.newaxis].T             
    n_feat_num = np_x.shape[1]  # the feature number of input      
    np_x_L2 = np.linalg.norm(np_x, axis = 1)
    np_x_L2 = np_x_L2[:, np.newaxis]
    np_x_L2 = np.tile(np_x_L2, (1,n_feat_num))
    np_x_L2_1 = 1.0/np_x_L2   
    np_x_L2Norm = np_x * np_x_L2_1    
    np_x_L2Norm[np.isnan(np_x_L2Norm)] = 0.0 # if value is nan and set 0.0   
    return np_x_L2Norm  
'''
get cos sim
'''
def get_cos_sim(np_A, np_B):
    return np.inner(np_A, np_B)/(np.linalg.norm(np_A) * np.linalg.norm(np_B))