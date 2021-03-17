# -*- coding: utf-8 -*-

import argparse
import cv2
import glob
import numpy as np
import os
import re

from ymd.img_utils.channel import (make_three_channels, merge_channels)

def test_directory(d):
    if ( not os.path.isdir(d) ):
        os.makedirs(d)

def find_files(d, p):
    '''
    Arguments:
    d (str): Directory to apply pattern p.
    p (str): Search pattern.

    Returns:
    A list of filenames.
    '''
    pattern = os.path.join(d, p)
    files = sorted( glob.glob( pattern, recursive=True ) )
    assert( len(files) > 0 ), 'No files found by {}'.format(pattern)
    return files

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        '%s does not exist. ' % ( fn )
    img = cv2.imread( fn, cv2.IMREAD_UNCHANGED )

    return img

def extract_name_from_fn(fn):
    name = os.path.basename(fn)
    m = re.search(r'(\d+)-(\d+)-(\d+)', fn)
    assert( m is not None ), '{} does not contain {}'.format(fn, r'(\d+)-(\d+)-(\d+)')
    return m.group()

def read_and_compose_grid(files):
    N = len(files) // 3

    # Read the first image.
    img = read_image(files[0])
    H, W = img.shape[:2]

    # The canvas.
    canvas = np.zeros( ( H * N, 4 * W, 3 ), dtype=np.uint8 )
    
    for i in range( N ):
        idx = i * 3

        img0 = read_image( files[idx] )
        img1 = read_image( files[idx+1] )
        img2 = read_image( files[idx+2] )

        img0_3C = make_three_channels(img0)
        img1_3C = make_three_channels(img1)
        img2_3C = make_three_channels(img2)

        hStart = i * H
        
        canvas[ hStart:hStart+H,   0:W,   ... ] = img0_3C
        canvas[ hStart:hStart+H,   W:2*W, ... ] = img1_3C
        canvas[ hStart:hStart+H, 2*W:3*W, ... ] = img2_3C

        # Merge to a single RGB image.
        canvas[ hStart:hStart+H, 3*W:4*W, ... ] = \
            merge_channels( ( img2, img0, img1 ) ) # The image will be written in BGR order.

        # The text.
        caseName = extract_name_from_fn(files[idx])
        _, textH = cv2.getTextSize( caseName, cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, thickness=1 )
        cv2.putText( canvas, caseName, ( 0, int(hStart+2.1*textH) ), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, color=( 0, 0, 255 ), thickness=1)

    return canvas

def handle_args():
    parser = argparse.ArgumentParser(description='Generate image grid. ')

    parser.add_argument('imagedir', type=str, 
        help='The directory contains all the images. ')

    parser.add_argument('outdir', type=str, 
        help='The output directory. ')

    parser.add_argument('outname', type=str, 
        help='The filename of the output under outdir. ')

    parser.add_argument('--image-pattern', type=str, default='**/*_压伤.jpg', 
        help='The search pattern for the images. ')

    parser.add_argument('--debug', action='store_true', default=False, 
        help='Set this flag to enable debug mode. ')

    return parser.parse_args()

def main():
    # Handle the arguments.
    args = handle_args()

    # Prepare the output directory.
    test_directory(args.outdir)

    # Find the files.
    imageFnList = find_files( args.imagedir, args.image_pattern )

    assert( len(imageFnList) % 3 == 0 ), \
        'The number of images must be a multiple of 3. len(imageFnList) = {}. '.format(len(imageFnList))

    print( '%d image files found' % ( len(imageFnList) ) )
    print('Generate the grid...')

    canvas = read_and_compose_grid(imageFnList)

    # Save.
    cv2.imwrite( os.path.join(args.outdir, args.outname), canvas )

    return 0

if __name__ == '__main__':
    import sys
    print('Hello, %s! ' % ( os.path.basename(__file__) ))
    sys.exit( main() )
