# coding: utf-8

# Author: Yaoyu Hu <yyhu_live@outlook.com>

import cv2
import numpy as np

def make_three_channels(img):
    assert( img.ndim == 2 ), \
        f'Wrong img shape: {img.shape}. Expecting to be 2 dimensional. '

    img = np.expand_dims(img, axis=-1)
    return np.tile( img, ( 1, 1, 3 ) )

def copy_split(img):
    if ( img.ndim == 1 ):
        return img.copy()
    
    return [ img[:, :, i].copy() for i in range(img.shape[2]) ]

def merge_channels(channels):
    '''
    This function tries to merge the arrays in channels into a new 
    multi-channel image. The input arrays must have single channel.

    If there is less or equal 3 cannels, the arrays will be
    copied into the RGB channels in the order of their indices. If there is 
    more than 3 channels, then an exception will be raised.

    This funciton always return a 3 chennel image with dtype the same as
    the first array in channels.

    The order of RGB channels are hanled by the user.

    Arguments:
    channels (list of arrays): The channels.

    Returns:
    A merged image.
    '''

    N = len(channels)
    assert( 0 < N <= 3 ), f'Wrong number of channels: {N}. '

    # Get the shape of the input.
    H, W = channels[0].shape[:2]

    # The output image.
    img = np.zeros( ( H, W, 3 ), dtype=channels[0].dtype )

    # Fill values in img.
    for i in range(N):
        img[:, :, i] = channels[i] # This makes a copy.

    return img