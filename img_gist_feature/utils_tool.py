# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import shutil
import imghdr

# @创建文件夹
# @input:s_path 保存的文件夹
# @output:
def recur_mkdir(s_path, run_logger=None):  
    if os.path.exists(s_path) and os.path.isdir(s_path):
        run_logger and run_logger.info("%s has been created" % s_path)
    else:
        os.makedirs(s_path)
        run_logger and run_logger.info("%s create" % s_path)

# 删除一个文件
def rm_file(s_file):
     try: 
         os.remove(s_file) # 旧文件删除
     except Exception as e:
         print('Err: cant'' remove %s, %s' % (s_file, str(e)), 'err')
         return -1
     return 0




# 复制一个文件
def cp_file(s_src_file, s_dest_file):
    try: 
        shutil.copyfile(s_src_file, s_dest_file) # 旧文件复制到临时文件夹中 
    except Exception as e:
        print('Err: cant''t copy %s to %s, %s' % (s_src_file, s_dest_file, str(e)))
        return -1
    return 0