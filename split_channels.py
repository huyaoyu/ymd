
import argparse
import cv2
from multiprocessing import Pool
import numpy as np
import os
import sys

from CommonPython.Filesystem import Filesystem

from ymd import img_utils

def find_files(d, pattern):
    files = Filesystem.find_files(d, pattern)

    if ( 0 == len(files) ):
        ss = '%s/**/%s' % (d, pattern)
        raise Exception('No files found with %s. ' % ( ss ))

    return files

def compose_output_fn(fn):
    parts = Filesystem.get_filename_parts(fn)
    return os.path.join( parts[0], f'{parts[1]}_BRG.png' )

def process_single_image(fn):
    print(fn)

    # Read the image.
    img = img_utils.io.read_image(fn)

    # Split the channels.
    # This is in BGR order.
    bgr = img_utils.channel.copy_split(img)

    # Get the image shape.
    H, W = img.shape[:2]

    # Create a image row.
    canvas = np.zeros( (H, 4*W, 3), dtype=np.uint8 )

    # Fill in values.
    canvas[:,   0:W, ...] = img
    canvas[:,   W:2*W, 0] = bgr[0]
    canvas[:, 2*W:3*W, 2] = bgr[2]
    canvas[:, 3*W:4*W, 1] = bgr[1]

    # Compose the output filename.
    outFn = compose_output_fn(fn)

    # Save.
    cv2.imwrite(outFn, canvas)

def single_process(args):
    process_single_image(*args)

def handle_arguments():
    parser = argparse.ArgumentParser(description='Split channels. ')

    parser.add_argument('indir', type=str, 
        help='The input directory. ')

    parser.add_argument('--pattern', type=str, default='*.jpg',
        help='The search pattern for the input images. ')

    parser.add_argument('--np', type=int, default=2, 
        help='The number of processes. ')

    args = parser.parse_args()

    return args

def main():
    print('Hello, %s! ' % (os.path.basename(__file__)))

    args = handle_arguments()

    files = find_files( args.indir, args.pattern )

    poolArgs = [ [fn] for fn in files ]

    with Pool(args.np) as p:
        p.map( single_process, poolArgs )

    return 0

if __name__ == '__main__':
    sys.exit( main() )

