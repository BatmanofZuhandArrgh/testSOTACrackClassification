import cv2
import os
import glob as glob
import numpy as np

if __name__ == '__main__':
    for img_path in glob.glob('datasets/AEL/gt/LCMS/*.png'):
        img = cv2.imread(img_path) 
    # img = cv2.imread('datasets/AEL/img/AIGLE_RN/Im_GT_AIGLE_RN_C18bor.jpg')
        print(img.shape)
        print(np.unique(img,return_counts=True))