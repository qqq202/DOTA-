png_file = r'G:\DOTA_qie_v2\train\images'
jpg_file = r'G:\DOTA_qie_v2\train\jpg\\'

import os
import cv2
import numpy as np
from PIL import Image

filepath = png_file
filename = os.listdir(filepath)
base_dir = filepath + "\\"

for img in filename:
    '''修改图像后缀名'''
    if os.path.splitext(img)[1] == '.png' or '.PNG':
        name = os.path.splitext(img)[0]
        newFileName = name + ".jpg"
    im = Image.open(base_dir + img)
    im.save(jpg_file + newFileName)
    im_gray1 = np.array(im)
    # cv2.imwrite(jpg_file + newFileName, im_gray1)