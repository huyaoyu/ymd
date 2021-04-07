# coding=utf-8

import argparse
import cv2
import glob
from multiprocessing import Pool
from multiprocessing import shared_memory
import numpy as np
import os
import pandas as pd
import time

from ymd.img_utils.io import read_image
from ymd.img_utils.binary import binarize
from ymd.align import HomographyCpu

def get_filename_parts(fn):
    s0 = os.path.split(fn)
    s1 = os.path.splitext(s0[1])
    
    if ( s0[0] == '' ):
        s0[0] = '.'

    return ( s0[0], *s1 )

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

def initialize_shared_memory(img):
    shm = shared_memory.SharedMemory( 
        create=True, size=np.prod( img.shape ) )
    shm.buf[:] = img.reshape((-1, ))
    return shm

def restore_image_from_shm(shm, shape):
    return np.ndarray( 
        np.prod(shape), dtype=np.uint8, buffer=shm.buf 
        ).reshape(shape)

def blur_resize_shape(img, newShape, kenerlSize=(5, 5)):
    imgG = cv2.GaussianBlur( img, kenerlSize, 0 )
    return cv2.resize(imgG, newShape, interpolation=cv2.INTER_CUBIC)

def blur_resize_by_width(img, newW):
    newH = int( np.ceil( newW / img.shape[1] * img.shape[0] ) )
    return blur_resize_shape( img, ( newW, newH ) )

def merge_two_single_channels(ref, tst):
    # Convert ref to a 3-channel grayscale image.
    ref = np.tile( np.expand_dims(ref, axis=-1), (1, 1, 3) )

    # Convert tst to a green image.
    g = np.zeros_like( ref )
    g[:, :, 1] = tst

    # Merge. 
    return cv2.addWeighted( ref, 0.7, g, 0.3, 0 )

def compose_filename(outDir, fn, suffix='', ext='.png'):
    parts = get_filename_parts(fn)
    return os.path.join( outDir, '%s%s%s' % ( parts[1], suffix, ext ) )

def process_single(shmName, shape, hgWidth, fn, outDir, imgExt='.png'):
    timeStart = time.time()

    # Show the input filename.
    print(fn)

    # Get the reference/destination image from shared memory.
    shm = shared_memory.SharedMemory( name=shmName )
    dstImg = restore_image_from_shm(shm, shape)

    # Read the test/source image.
    try:
        srcImg = read_image(fn)

        if ( dstImg.shape != srcImg.shape ):
            print(f'dstImg and srcImg {fn} have different shapes. dstImg.shape = {dstImg.shape}, srdImg.shape = {srcImg.shape}. ' )

        # Resize.
        dstImgR = blur_resize_by_width( dstImg, hgWidth )
        srcImgR = blur_resize_by_width( srcImg, hgWidth )

        # Binarize.
        dstImgR, srcImgR = binarize( (dstImgR, srcImgR), thres=50, gf=False )

        # Create a homography computer.
        hg = HomographyCpu()

        # Compute the homography.
        # FXM stands for "feature extraction and matching".
        # HG stands for "homography".
        retFlag, hMat, goodMatches, nHomographyMatched, diff, timeFXM, timeHG = \
            hg( dstImgR, srcImgR )

        if (retFlag):
            # Scale the homography matrix back to original scale.
            hMat = HomographyCpu.scale_homography_matrix( 
                hMat, dstImgR.shape, dstImg.shape, srcImgR.shape, srcImg.shape )

            # Warp the test/source image.
            warped = cv2.warpPerspective(
                srcImg, hMat, ( shape[1], shape[0] ), flags=cv2.INTER_LINEAR)

            # Merge the reference/destination and test/source images.
            merged = merge_two_single_channels( dstImg, warped )

            # Annotate the merged image.
            annText = 'N: %d/%d(%.2f), D: %.2fpix, FXM: %3.1fms, HG: %3.1fms' % \
                ( nHomographyMatched, len(goodMatches), nHomographyMatched/len(goodMatches), diff, timeFXM*1000, timeHG*1000 )
            (textW, textH), _ = cv2.getTextSize( annText, cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=2, thickness=2 )
            cv2.putText( merged, annText, ( int(0.1*textW), int(2.5*textH) ), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=2, color=( 0, 0, 255 ), thickness=2)

            # Write the merged image to the output directory.
            outFn = compose_filename( outDir, fn, '_Merged', ext=imgExt )
            cv2.imwrite( outFn, merged )

            # Write the statistics.
            outFn = compose_filename( 
                os.path.join( outDir, 'stat' ), fn, ext='.csv' )
            np.savetxt( outFn, 
                [ nHomographyMatched, 
                len(goodMatches), 
                nHomographyMatched/len(goodMatches), 
                diff, 
                timeFXM, timeHG ], fmt='%f', delimiter=',' )
        else:
            raise Exception('Homography computation failed. ')
    except Exception as e:
        # Write the statistics.
        outFn = compose_filename( 
            os.path.join( outDir, 'stat' ), fn, ext='.csv' )
        np.savetxt( outFn, 
            [ -1, 
            -1, 
            -1, 
            -1, 
            -1, -1 ], fmt='%f', delimiter=',' )

    # Clean up the shared memory.
    shm.close()

def pooled_single(args):
    process_single(*args)

def merge_stats(fnList):
    stacked = []
    for fn in fnList:
        # Read the content.
        stat = np.loadtxt( fn, delimiter=',', dtype=np.float32 )
        stacked.append( stat )

    return np.stack( (stacked), axis=0 )

def write_stats(fn, fnList, stats):
    assert( len(fnList) == stats.shape[0] ), \
        f'len(fnList) = {len(fnList)}, stats.shape[0] = {stats.shape[0]}'

    # Create a dataframe.
    df = pd.DataFrame.from_dict( {
        'source': fnList, 
        'homography matches': stats[:, 0],
        'good matches': stats[:, 1],
        'match ratio': stats[:, 2],
        'reprojection error (pixel)': stats[:, 3],
        'feature extraction and matching time (s)': stats[:, 4], 
        'homography time (s)': stats[:, 5],
    } )

    # Save the dataframe.
    df.to_csv( fn, index=False )

def handle_args():
    parser = argparse.ArgumentParser(description='Align images. ')

    parser.add_argument('imagedir', type=str, 
        help='The directory contains all the images. ')

    parser.add_argument('outdir', type=str, 
        help='The output directory. ')

    parser.add_argument('--image-pattern', type=str, default='*.jpg', 
        help='The search pattern for the images. ')

    parser.add_argument('--hg-width', type=int, default=512, 
        help='The width of the intermediate image for homography computation. ')

    parser.add_argument('--image-write-ext', type=str, default='.jpg', 
        help='Set this flag to wite compressed JPEG.')

    parser.add_argument('--np', type=int, default=2, \
        help='The process number. ')

    parser.add_argument('--debug', type=int, default=0, 
        help='Use a positive integer to enable single process debug mode. ')

    return parser.parse_args()

def main():
    # Handle the arguments.
    args = handle_args()

    # Prepare the output directory.
    test_directory(args.outdir)
    statDir = os.path.join( args.outdir, 'stat' )
    test_directory( statDir )

    # Find the files.
    imageFnList = find_files( args.imagedir, args.image_pattern )
    nImages = len( imageFnList )
    assert( nImages > 1 ), f'Not enough images. Need at leat 2 images to work. '

    # Read the first image.
    img = read_image( imageFnList[0] )

    # Initialize a shared memory object.
    shm = initialize_shared_memory(img)

    if ( args.debug > 0 ):
        # Figure out the maximum image number. 
        nImages = min( args.debug, nImages )
        for i in range( 1, nImages ):
            fn = imageFnList[i]
            process_single( shm.name, img.shape, args.hg_width, fn, args.outdir, args.image_write_ext )
    else:
        # Prepare the arguments.
        poolArgs = [ 
            [ shm.name, img.shape, args.hg_width, imageFnList[i], args.outdir, args.image_write_ext ] 
            for i in range( 1, nImages ) ]
        
        with Pool(args.np) as p:
            p.map( pooled_single, poolArgs )

    print(f'{nImages} in total. ')

    # Gather the statistics.
    statFnList = find_files( statDir, '*.csv' )
    assert( len(statFnList) == nImages - 1 ), \
        f'Expect len(statFnList) == nImages. len(statFnList) = {len(statFnList)}, nImages = {nImages}.'

    stats = merge_stats(statFnList)
    
    # Write the statistics to the filesystem.
    write_stats( os.path.join( args.outdir, 'MergedStats.csv' ), 
        imageFnList[1:nImages], stats )

    # Clean up the shared memory.
    shm.close()
    shm.unlink()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit( main() )