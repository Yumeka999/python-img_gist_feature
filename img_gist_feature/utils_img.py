# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import imghdr


''' Is numpy matrix all no-zero data in alpha channel ''' 
def is_single_alpha(np_raw_img):
    if(np_raw_img.shape[-1]!=4):
        return False
    for i in range(3):
        if(sum(sum(np_raw_img[:,:,i]>0)))!=0:
            return False
    return True

''' Convert raw image to small gray image, resize is  n_resize * n_resize ''' 
def img_2gray(np_img_raw):
    if np_img_raw is None:
        return None, -3
    # Raw Image is BGR imge, so convert rgb to gray
    np_img_gray = None
    if len(np_img_raw.shape) == 3 and np_img_raw.shape[2] == 3:
        np_img_gray = cv2.cvtColor(np_img_raw, cv2.COLOR_BGR2GRAY)
    # Raw Image is BGRA imge, there are different situation to solve
    elif len(np_img_raw.shape) == 3 and np_img_raw.shape[2] == 4:
        n_sence = 3
        np_img_gray_choose = np.zeros([np_img_raw.shape[0], np_img_raw.shape[1], n_sence], dtype=np.uint8)

        np_img_gray_choose[:, :, 0] = 255 - np_img_raw[:, :, 3]
        np_img_gray_choose[:, :, 1] = cv2.cvtColor(np_img_raw, cv2.COLOR_BGRA2GRAY)
        np_img_gray_choose[:, :, 2] = cv2.cvtColor(np_img_raw[:, :, 0:3], cv2.COLOR_BGR2GRAY)

        # Get nonzero element of every resize gray
        ln_sence_non0_num = []
        for i in range(n_sence):
            ln_sence_non0_num.append(len(np_img_gray_choose[:, :, i].nonzero()[0]))

        # Which image has most nonzero element
        if len(set(ln_sence_non0_num)) > 1:
            n_max_index = ln_sence_non0_num.index(max(ln_sence_non0_num))
            np_img_gray = np_img_gray_choose[:, :, n_max_index]
        else:
            # Which image has most different element
            ln_diff_pix_num = []
            for i in range(n_sence):
                ln_diff_pix_num.append(len(np.unique(np_img_gray_choose[:, :, i])))
            n_max_index = ln_diff_pix_num.index(max(ln_diff_pix_num))
            np_img_gray = np_img_gray_choose[:, :, n_max_index]
    # Raw Image is gray image
    elif len(np_img_raw.shape) == 3 and np_img_raw.shape[2] == 1:
        np_img_gray = np_img_raw[:, :, 0]
    elif len(np_img_raw.shape) == 2:
        np_img_gray = np_img_raw

    #    print(np_img_gray.shape)
    return np_img_gray, 0
    
'''  a image to bgr format '''   
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

''' resize a image '''     
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



''' A image deblank '''
def img_deblank(np_img_raw):
    # only gray and color image can be deblanked
    n_shape_size = len(np_img_raw.shape)
    if n_shape_size < 2 or n_shape_size > 4 or (n_shape_size == 3 and np_img_raw.shape[2] !=3 ):
        return None, -1

    # gray image strategy
    if n_shape_size == 2:
        n_row, n_col = np_img_raw.shape
        # OTSU to get binary image
        try:
            thrsh, np_img_otsu = cv2.threshold(np_img_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception as e:
            s_run_msg = 'url cv threshold otsu error, err:%s' % (str(e))
            return None, -1

        # find coordinate of balck point of left-top
        np_blank_index = np.argwhere(np_img_otsu == 0)

        # if no black in small image, so don't need to do deblank
        if np_blank_index.shape[0] == 0:
            return np_img_raw, 1

        # Get coordinate of left-top  right-bottom
        n_row_min = np.min(np_blank_index[:, 0])
        n_row_max = np.max(np_blank_index[:, 0])
        n_col_min = np.min(np_blank_index[:, 1])
        n_col_max = np.max(np_blank_index[:, 1])

        # if no blank so don't need to do deblank
        if n_row_min == 0 and n_col_min == 0 and n_row_max == n_row and n_col_max == n_col:
            return np_img_otsu, 1

        # get deblank zone of small image
        np_img_deblank_zone = np_img_raw[n_row_min: n_row_max + 1, n_col_min:n_col_max + 1]
        return np_img_deblank_zone, 0
    else:
        row, col, c = np_img_raw.shape

        tempr0 = 0
        tempr1 = 0
        tempc0 = 0
        tempc1 = 0

        for r in range(row):
            if np_img_raw[r,:,:].sum() != 765 * col:
                tempr0 = r
                break

        for r in range(row-1,0,-1):
            if np_img_raw[r,:,:].sum() != 765 * col:
                tempr1 = r
                break

        for c in range(col):
            if np_img_raw[:,c,:].sum() != 765 * row:
                tempc0=c
                break

        for c in range(col-1, 0,-1):
            if np_img_raw[:,c,:].sum() != 765 * row:
                tempc1=c
                break

        np_img_deblank_zone = np_img_raw[tempr0:tempr1+1, tempc0:tempc1+1,:]
        return np_img_deblank_zone, 0


def get_all_frame_from_gif(s_gif_url):
    return 