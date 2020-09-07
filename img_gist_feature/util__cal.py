# -*- coding: utf-8 -*-

import os
import time
import numpy as np

# Get all cos sim
#@numba.autojit
def get_all_cos_sim(np_A ,np_B, np_B_L2 = None, run_logger=None, b_print=False):
    n_num = np_B.shape[0]
    
    t1 = time.time()  
    np_A_L2 = np.linalg.norm(np_A)
    np_A_L2 = np.tile(np_A_L2, (1, n_num))   
    t2 = time.time()

    s_msg = "A L2 time %.3f" % (t2-t1)
    run_logger and run_logger.info(s_msg)
    b_print and print(s_msg)
    
    if np_B_L2 is None:
        t1 = time.time()  
        np_B_L2 = np.linalg.norm(np_B, axis = 1)
        t2 = time.time()

        s_msg = "B L2 time %.3f" % (t2-t1)
        run_logger and run_logger.info(s_msg)
        b_print and print(s_msg)

    t1 = time.time()
    np_inner = np_A.dot(np_B.T)
    
    np_cos_sim = np_inner/(np_A_L2 * np_B_L2)
    np_cos_sim = 0.5 + 0.5 * np_cos_sim
    t2 = time.time()

    s_msg = "cos time %.3f" % (t2-t1)
    run_logger and run_logger.info(s_msg)
    b_print and print(s_msg)

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
    
    np_x_L2Norm = np_x * np_x_L2_1
      
    np_where_are_nans = np.isnan(np_x_L2Norm) # if value is nan and set 0.0
    np_x_L2Norm[np_where_are_nans] = 0.0
    
    return np_x_L2Norm


def get_cos_sim(np_A, np_B):
    return np.inner(np_A, np_B)/(np.linalg.norm(np_A) * np.linalg.norm(np_B))