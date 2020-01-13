# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import imghdr

# Is numpy matrix all no-zero data in alpha channel
def is_single_alpha(np_raw_img):
    if(np_raw_img.shape[-1]!=4):
        return False
    for i in range(3):
        if(sum(sum(np_raw_img[:,:,i]>0)))!=0:
            return False
    return True

def img_2bgr(np_img_in):
    if np_img_in is None:
        return None, -3
    
    # if raw image is uint16 so conver to uint8 
    np_img_bgr = None
    if len(np_img_in.shape) == 3 and np_img_in.shape[2] == 3: # Raw Image is BGR imge, so continue
        np_img_bgr = np_img_in
    elif len(np_img_in.shape) == 3 and np_img_in.shape[2] == 4: # Raw Image is BGRA imge, there are different situation to solve
        h, w, c = np_img_in.shape

        np_img_bgr_1 = cv2.cvtColor(np_img_in, cv2.COLOR_BGRA2BGR)
       
        b, g, r, a = cv2.split(np_img_in)
        b = cv2.convertScaleAbs(b, alpha=(255.0/65535.0)) # (b/256).astype('uint8')
        g = cv2.convertScaleAbs(g, alpha=(255.0/65535.0))
        r = cv2.convertScaleAbs(r, alpha=(255.0/65535.0))
        a = cv2.convertScaleAbs(a, alpha=(255.0/65535.0))
        new_img  = cv2.merge((b, g, r))
        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR) 
        new_img = cv2.bitwise_and(new_img, new_img, mask = a)
        np_img_bgr_2 = cv2.add(new_img, not_a)

        # Which image has most not white
        np_img_gray_1 = cv2.cvtColor(cv2.convertScaleAbs(np_img_bgr_1, alpha=(255.0/65535.0)), cv2.COLOR_BGR2GRAY)
        np_img_gray_2 = cv2.cvtColor(np_img_bgr_2, cv2.COLOR_BGR2GRAY)

        n_info_1 = len(np.unique(np_img_gray_1))
        n_info_2 = len(np.unique(np_img_gray_2))    
        if n_info_1 >= n_info_2:
            np_img_bgr = np_img_bgr_1
        else:
            np_img_bgr = np_img_bgr_2
    elif len(np_img_in.shape) == 3 and np_img_in.shape[2] == 1: # Raw Image is gray image
        np_img_bgr = np.tile(np_img_in, (1, 1, 3))     # 256x256x1 ==> 256x256x3
    elif len(np_img_in.shape) == 2:
        np_img_bgr = np.tile(np_img_in, (3, 1, 1))     # 256x256 ==> 3x256x256
        np_img_bgr = np.transpose(np_img_bgr, (1, 2, 0))  # 3x256x256 ==> 256x256x3


    return np_img_bgr, 0
    
def img_resize(np_img_in, n_resize):
    np_img_resize = np_img_in
    n_row, n_col, n_chanel = np_img_resize.shape
    if n_resize > 0 and (n_row != n_resize or n_col != n_resize):
        try:
            np_img_resize = cv2.resize(np_img_resize, (n_resize, n_resize), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        except Exception as e:
            return None, -3

    return np_img_resize, 0


''' A image is usable image  '''
def is_usable_img(s_img_url):
    if not os.path.exists(s_img_url): return False
         
    if imghdr.what(s_img_url) is None: return False
            
    try:
        np_img_in = cv2.imdecode(np.fromfile(s_img_url, dtype=np.uint8),-1)
    except Exception as e:
        print('img url:%s, err:%s' % (s_img_url, str(e)))
        return False 

    if np_img_in is None:
        print('img url:%s, is null mat' % s_img_url)
        return False
    
    return True