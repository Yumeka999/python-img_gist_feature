# -*- coding: utf-8 -*-

import os
import sys
import shutil

S_NOW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(S_NOW_DIR)

'''
Creat Directory
'''
def recur_mkdir(s_path, run_log=None, b_print=False):  
    n_ret = 0
    if os.path.exists(s_path) and os.path.isdir(s_path):
        n_ret, s_msg = 1, "%s has been created" % s_path
    else:
        try:
            os.makedirs(s_path)
            s_msg = "%s create" % s_path
        except Exception as e:
            n_ret, s_msg = -1, "mkdir err: %s" % str(e)    
    run_log and run_log.warning(s_msg)
    b_print and print(s_msg)

    return n_ret

'''
Copy a file or directory
'''
def cp_file_dir(s_in_url, s_out_url, run_log=None, b_print=False):
    if os.path.exists(s_out_url):  
        if rm_file_dir(s_out_url) < 0:
            s_msg = 'Err in rm_file_dir'
            run_log and run_log.error(s_msg)
            b_print and print(s_msg)

            return -1
    cp_func = shutil.copyfile if os.path.isfile(s_in_url) else shutil.copytree

    try:
        cp_func(s_in_url, s_out_url) # Copy a old file to tmp dir 
        return 0     
    except Exception as e:
        s_msg = 'Err: cant''t copy %s to %s, %s' % (s_in_url, s_out_url, str(e))
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        
        return -1
   
'''      
Move a file or directory
''' 
def mv_file_dir(s_in_url, s_out_url, run_log=None, b_print=False):
    try: 
        shutil.move(s_in_url, s_out_url) # using shutil.move

        return 0
    except Exception as e:
        s_msg = 'Err: cant''t copy %s to %s, %s' % (s_in_url, s_out_url, str(e))
        run_log and run_log.error(s_msg)
        b_print and print(s_msg) 

        return -1

'''
Delete a file
'''
def rm_file_dir(s_in_url, run_log=None, b_print=False):
    try:
        if os.path.exists(s_in_url): # is a directory
            shutil.rmtree(s_in_url) 

            return 0
        else:
            s_msg = 'Err: not exists %s' % (s_in_url)
            run_log and run_log.error(s_msg)
            b_print and print(s_msg)

            return 1
    except Exception as e:
        s_msg = 'Err: cant'' remove %s, %s' % (s_in_url, str(e))
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)

        return -1

'''
Get right string
'''
def get_usable_str(s_in):
    s_tmp = s_in  
    s_unvalid = '<>,\/|,:.,''",*,?\t\r\n'
    for ch in s_unvalid:
        s_tmp = s_tmp.replace(ch,'')
    s_tmp = s_tmp.replace(u'\u3000','')
    
    return s_tmp

'''
time to millisecond
'''
def time_2_millsecond(s_time):
    ls_time = s_time.split(":")
    n_hour, n_min, n_second = int(ls_time[0]), int(ls_time[1]), int(ls_time[2])
    return 1000 * (3600 * n_hour + 60 * n_min + n_second)