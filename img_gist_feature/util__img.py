# -*- coding: utf-8 -*-
import os
import sys
import cv2
import imghdr
import imageio
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim
S_NOW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(S_NOW_DIR)
from util__base import *

''' 
Is numpy matrix all no-zero data in alpha channel 
''' 
def is_single_alpha(np_raw_img, run_log=None, b_print=False):
    try:
        if np_raw_img.shape[-1] != 4:  
            return False
        for i in range(3):
            if sum(sum(np_raw_img[:,:,i]>0)) != 0: 
                return False        
        return True
    except Exception as e:
        s_msg = "err: %s" % str(e)
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return False

''' 
Convert raw image to small gray image, resize is  n_resize * n_resize 
''' 
def img_2gray(np_img_raw, run_log=None, b_print=False):
    if np_img_raw is None:
        s_msg = "input image null"
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return None, -3
    np_img_gray = None # Raw Image is BGR imge, so convert rgb to gray
    if len(np_img_raw.shape) == 3 and np_img_raw.shape[2] == 3:
        np_img_gray = cv2.cvtColor(np_img_raw, cv2.COLOR_BGR2GRAY) 
    elif len(np_img_raw.shape) == 3 and np_img_raw.shape[2] == 4: # Raw Image is BGRA imge, there are different situation to solve
        n_sence = 3
        np_img_gray_choose = np.zeros([np_img_raw.shape[0], np_img_raw.shape[1], n_sence], dtype=np.uint8)

        np_img_gray_choose[:, :, 0] = 255 - np_img_raw[:, :, 3]
        np_img_gray_choose[:, :, 1] = cv2.cvtColor(np_img_raw, cv2.COLOR_BGRA2GRAY)
        np_img_gray_choose[:, :, 2] = cv2.cvtColor(np_img_raw[:, :, 0:3], cv2.COLOR_BGR2GRAY)
   
        ln_sence_non0_num = [] # Get nonzero element of every resize gray
        for i in range(n_sence):
            ln_sence_non0_num.append(len(np_img_gray_choose[:, :, i].nonzero()[0]))  
        if len(set(ln_sence_non0_num)) > 1: # Which image has most nonzero element
            n_max_index = ln_sence_non0_num.index(max(ln_sence_non0_num))
            np_img_gray = np_img_gray_choose[:, :, n_max_index]
        else: # Which image has most different element
            ln_diff_pix_num = []
            for i in range(n_sence):
                ln_diff_pix_num.append(len(np.unique(np_img_gray_choose[:, :, i])))
            n_max_index = ln_diff_pix_num.index(max(ln_diff_pix_num))
            np_img_gray = np_img_gray_choose[:, :, n_max_index] 
    elif len(np_img_raw.shape) == 3 and np_img_raw.shape[2] == 1: # Raw Image is gray image
        np_img_gray = np_img_raw[:, :, 0]
    elif len(np_img_raw.shape) == 2:
        np_img_gray = np_img_raw
    return np_img_gray, 0

'''  
a image to bgr format 
'''   
def img_2bgr(np_img_in, run_log=None, b_print=False):
    if np_img_in is None:
        return None, -3 

    np_img_bgr = None # if raw image is uint16 so conver to uint8 
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
        np_img_gray_1 = cv2.cvtColor(cv2.convertScaleAbs(np_img_bgr_1, alpha=(255.0/65535.0)), cv2.COLOR_BGR2GRAY) # Which image has most not white
        np_img_gray_2 = cv2.cvtColor(np_img_bgr_2, cv2.COLOR_BGR2GRAY)
        n_info_1 = len(np.unique(np_img_gray_1))
        n_info_2 = len(np.unique(np_img_gray_2))    
        np_img_bgr = np_img_bgr_1 if n_info_1 >= n_info_2 else np_img_bgr_2
    elif len(np_img_in.shape) == 3 and np_img_in.shape[2] == 1: # Raw Image is gray image
        np_img_bgr = np.tile(np_img_in, (1, 1, 3))     # 256x256x1 ==> 256x256x3
    elif len(np_img_in.shape) == 2:
        np_img_bgr = np.tile(np_img_in, (3, 1, 1))     # 256x256 ==> 3x256x256
        np_img_bgr = np.transpose(np_img_bgr, (1, 2, 0))  # 3x256x256 ==> 256x256x3
    return np_img_bgr, 0

''' 
resize a image 
'''     
def img_resize(np_img_in, ln_resize, run_log=None, b_print=False):
    try:
        np_img_resize = cv2.resize(np_img_in, ln_resize, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return np_img_resize, 0
    except Exception as e:
        s_msg = 'resize err:%s' % str(e)
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return None, -3
    
''' 
A image is usable image  
'''
def is_usable_img(s_img_url, run_log=None, b_print=False):
    if not os.path.exists(s_img_url): 
        s_msg = "not find %s" % s_img_url
        run_log and run_log.warning(s_msg)
        b_print and print(s_msg)
        return False
    if s_img_url.rfind('.gif') > 0: 
        return True
    if s_img_url.rfind('.bpg') > 0: 
        return True if is_bpg_img(s_img_url, run_log, b_print) == 0 else False

    try:
        np_img_in = cv2.imdecode(np.fromfile(s_img_url, dtype=np.uint8),-1)
    except Exception as e:
        s_msg = 'img url:%s, err:%s' % (s_img_url, str(e))
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return False

    if np_img_in is None:
        s_msg = 'img url:%s, is null mat' % s_img_url
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return False
    n_h, n_w = np_img_in.shape[0], np_img_in.shape[1]
    if n_h < 200 or n_w < 200: 
        s_msg = 'img url:%s, smart' % s_img_url
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return False  
    return True

''' 
A image deblank 
'''
def img_deblank(np_img_raw, run_log=None, b_print=False):
    # only gray and color image can be deblanked
    n_shape_size = len(np_img_raw.shape)
    if n_shape_size < 2 or n_shape_size > 4 or (n_shape_size == 3 and np_img_raw.shape[2] !=3 ):
        return None, -1
    # gray image strategy
    if n_shape_size == 2:
        n_row, n_col = np_img_raw.shape
        try: # OTSU to get binary image
            thrsh, np_img_otsu = cv2.threshold(np_img_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception as e:
            s_msg = 'url cv threshold otsu error, err:%s' % (str(e))
            run_log and run_log.error(s_msg)
            b_print and print(s_msg)
            return None, -1
        np_blank_index = np.argwhere(np_img_otsu == 0) # find coordinate of balck point of left-top
        if np_blank_index.shape[0] == 0: # if no black in small image, so don't need to do deblank
            return np_img_raw, 1
        n_row_min = np.min(np_blank_index[:, 0]) # Get coordinate of left-top  right-bottom
        n_row_max = np.max(np_blank_index[:, 0])
        n_col_min = np.min(np_blank_index[:, 1])
        n_col_max = np.max(np_blank_index[:, 1]) 
        if n_row_min == 0 and n_col_min == 0 and n_row_max == n_row and n_col_max == n_col: # if no blank so don't need to do deblank
            return np_img_otsu, 1
        np_img_deblank_zone = np_img_raw[n_row_min: n_row_max + 1, n_col_min:n_col_max + 1] # get deblank zone of small image
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

'''
Get all frame form gif 
Mode
1             1 bit pixel, black and white, save 8 bit pixel
L             8 bit pixel, black and white
P             8 bit pixel, mapping other mode with using color palette
RGB           3×8 bit pixel, true color
RGBA          4×8位像素，真彩+透明通道
CMYK          4×8位像素，颜色隔离
YCbCr         3×8位像素，彩色视频格式
I             32位整型像素
F             32位浮点型像素
''' 
def get_all_frame_from_gif(s_gif_url, s_all_frame_out_dor, b_rgb=False, run_log=None, b_print=False):
    try:
        recur_mkdir(s_all_frame_out_dor, run_log)
        f_duration, n_frame_num = 0.0, 0
        pil_gif = Image.open(s_gif_url)
        b_animate = pil_gif.is_animated
        n_frame_num = pil_gif.n_frames

        if not b_animate:  
            return 1, 0.0 # static gif  
        for i in range(n_frame_num):
            pil_gif.seek(i)
            f_duration += pil_gif.info['duration']
            pil_sav = pil_gif
            if b_rgb and pil_gif.mode == "P":  
                pil_sav = pil_gif.convert("RGB")
            elif b_rgb and pil_gif.mode == "RGBA":
                pil_sav = pil_gif.convert("RGB") 
            pil_sav.save(os.path.join(s_all_frame_out_dor, "%s.png" % str(i+1).zfill(3)))
        return 0, n_frame_num / f_duration * 1000
    except Exception as e:
        s_msg = 'Err %s' % (str(e))
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return -1, 0.0

'''
get a gif frome each frame image 
'''
def gen_gif_from_frames(ls_img_path, s_gif_path, f_fps=0.06, run_log=None, b_print=False):
    try:
        l_img = []
        for e in ls_img_path:
            l_img.append(imageio.imread(e))
        imageio.mimsave(s_gif_path, l_img, 'GIF', duration=f_fps)
        return 0
    except Exception as e:
        s_msg = "err:%s" % str(e)
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return -1

'''
Get real format of a image
'''
def get_img_obv_and_true_ext(s_img_in_url, run_log=None, b_print=False):
    _, s_obv_ext = os.path.splitext(s_img_in_url)  # Get extension name of a image
    if not os.path.exists(s_img_in_url) or not os.path.isfile(s_img_in_url):
        s_msg = "%s not exists or not a file" % s_img_in_url
        run_log and run_log.warning(s_msg)
        b_print and print(s_msg)
        return s_obv_ext, ""
    s_true_ext = imghdr.what(s_img_in_url) 
    if s_true_ext is None:
        s_msg = "%s not a iamge with imghdr" % s_img_in_url
        run_log and run_log.warning(s_msg)
        b_print and print(s_msg) 
        return s_obv_ext, s_obv_ext
    else:
        return s_obv_ext, "." + s_true_ext

'''
Get numpy array from path of image
'''
def read_img(s_img_in_url, run_log=None, b_print=False):
    try:
        np_img = cv2.imdecode(np.fromfile(s_img_in_url, dtype=np.uint8), cv2.IMREAD_UNCHANGED)    
        return np_img
    except Exception as e:
        s_msg = 'Err %s' % str(e)
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return None
    
'''
Write a image with url
'''
def write_img(s_img_out_url, np_img, run_log=None, b_print=False):
    try:
        s_ext = s_img_out_url[s_img_out_url.rfind("."):]
        if s_ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"]:
            return -1
        cv2.imencode(s_ext, np_img)[1].tofile(s_img_out_url)
        return 0
    except Exception as e:
        run_log and run_log.error('Err %s' % str(e))
        return -1

'''
Resize a image in window size
'''
def img_resize_win(np_img_in, n_max, n_limit_ratio, run_log=None, b_print=False):
    n_h, n_w = np_img_in.shape[0], np_img_in.shape[1]
    re_h, re_w = 0, 0
    b_need_resize = True
    if n_h/n_w > n_limit_ratio or n_w/n_h > n_limit_ratio:  # If weight/hight > limit_ratio or hight/weight > limit_ratio
        b_need_resize = False   
    elif n_w <= n_max and n_h <= n_max: 
        b_need_resize = False   
    elif n_w > n_max and n_h <= n_max:  # if weight > max 
        re_w = n_max 
        re_h = (n_h*re_w)//n_w        
    elif n_w <= n_max and n_h >= n_max:  # if height > max
        re_h = n_max     
        re_w = (n_w*re_h)//n_h          
    else:
        re_w = n_max
        re_h = (n_h*re_w)//n_w
        if re_h > n_max:
            re_h = n_max     
            re_w = (n_w*re_h)//n_h   
    np_img_resize = cv2.resize(np_img_in, (re_w, re_h), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) if b_need_resize else np_img_in   
    return np_img_resize

'''
A image is right bpg format?
'''
def is_bpg_img(s_img_in_url, run_log=None, b_print=False):
    if not os.path.exists(s_img_in_url) or not os.path.isfile(s_img_in_url):
        s_msg = "the bpg of %s not exists" % s_img_in_url
        run_log and run_log.warning(s_msg)
        b_print and print(s_msg) 
        return -1
    if not s_img_in_url.endswith(".bpg"):
        s_msg = "not endwith .bpg"
        run_log and run_log.warning(s_msg)
        b_print and print(s_msg)
        return 1
    with open(s_img_in_url, "rb") as fp:
        for now in fp:
            n_byte_1, n_byte_2 = now[0], now[1] 
            if n_byte_1 == 0x42 and n_byte_2 == 0x50:
                return 0
            break
    s_msg = "not right bpg"
    run_log and run_log.warning(s_msg)
    b_print and print(s_msg)
    return -1

'''
Get histogram equalization image
''' 
def get_histeq_img(np_img_in, run_log=None, b_print=False):
    try:   
        n_shape_len, np_img_out = len(np_img_in.shape), None
        if n_shape_len == 2: # Single chanle image
            np_img_out = cv2.equalizeHist(np_img_in)
        elif n_shape_len == 3 and np_img_in.shape[2] == 3: # RGB three chanle
            np_img_out = np.zeros(np_img_in.shape)
            np_img_out[:,:,0] = cv2.equalizeHist(np_img_in[:,:,0])
            np_img_out[:,:,1] = cv2.equalizeHist(np_img_in[:,:,1])
            np_img_out[:,:,2] = cv2.equalizeHist(np_img_in[:,:,2])
        return np_img_out
    except Exception as e:
        s_msg = "err:%s" % str(e)
        run_log and run_log.warning()
        b_print and print(s_msg)
        return None

'''
Compute ssim
'''
def get_ssim(np_img_A, np_img_B, run_log=None, b_print=False):
    try:
        np_img_gray_A, n_ret_A = img_2gray(np_img_A, b_print=b_print)
        np_img_gray_B, n_ret_B = img_2gray(np_img_B, b_print=b_print)
        if n_ret_A !=0 or n_ret_B != 0:     # Must gray image
            s_msg = "error in img_2gray()"
            run_log and run_log.erro(s_msg)
            b_print and print(s_msg)
            return -1.0
        sim, _ = compare_ssim(np_img_A[:,:,0], np_img_B[:,:,0], full=True)   
        return sim
    except Exception as e:
        s_msg = 'err:%s' % str(e)
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return -1.0

'''
Canny edge detect
'''
def canny_edge_detect(np_img, n_low = 60 , n_high = 180, run_log=None, b_print=False):
    try:
        np_gray, n_ret = img_2gray(np_img)
        if np_gray is None:
            s_msg = "err in img_2gray"
            run_log and run_log.error(s_msg)
            b_print and print(s_msg)        
            return None               
        np_detect_edge = cv2.GaussianBlur(np_gray, (3, 3), 0)
        np_detect_edge = cv2.Canny(np_detect_edge, n_low, n_high)
        np_canny = cv2.bitwise_and(np_img, np_img, mask = np_detect_edge)  # just add some colours to edges from original image.    
        return np_canny
    except Exception as e:
        s_msg = 'err %s' % str(e)
        run_log and run_log.error(s_msg)
        b_print and print(s_msg)
        return None