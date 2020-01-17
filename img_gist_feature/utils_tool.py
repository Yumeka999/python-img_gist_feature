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