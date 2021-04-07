# coding: utf-8

# Author: Yaoyu Hu <yyhu_live@outlook.com>

import cv2

def binarize(imgs, thres=50, gf=True):
    '''
    Arguments:
    imgs (list of NumPy arrays): The images.
    thres (float): The binarization threshold.
    gf (bool): Set True to perform a Gaussian blur before binarizing.

    Returns:
    A list of binarized images.
    '''
    
    if ( isinstance(imgs, (list, tuple)) ):
        binarized = []

        for img in imgs:
            imgG    = cv2.GaussianBlur( img, (3, 3), 0 ) if gf else img
            _, imgB = cv2.threshold( imgG, thres, 255, cv2.THRESH_BINARY )

            binarized.append( imgB )

        return binarized
    else:
        imgG    = cv2.GaussianBlur( imgs, (3, 3), 0 ) if gf else imgs
        _, imgB = cv2.threshold( imgG, thres, 255, cv2.THRESH_BINARY )
        return imgB