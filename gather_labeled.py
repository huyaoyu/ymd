# -*- coding: utf-8 -*-

import argparse
import cv2
import glob
from multiprocessing import Pool
import numpy as np
import os
import re
import xml.etree.ElementTree as ET

from ymd.img_utils.binary import binarize
from ymd.align import HomographyTransform

from name_map import NAME_MAP

caseSearcher = re.compile(r'(\d+)-(\d+)')

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

def find_case_name_from_filename(fn):
    stem = os.path.splitext( os.path.basename(fn) )[0]

    # Get the special part from the stem.
    m = caseSearcher.search(stem)
    if ( m is None ):
        return None

    return m.group()

def convert_image_filenames_2_dict(fnList):
    d = dict()
    for fn in fnList:
        caseName = find_case_name_from_filename(fn)

        if ( caseName is None ):
            continue

        # Found a valid case name.
        if ( caseName in d.keys() ):
            d[caseName].append( fn )
        else:
            d[caseName] = [ fn ]

    return d

def compute_padding(x, t):
    assert( t > x)
    m = t - x
    p0 = m//2
    p1 = m - p0
    return p0, p1

class BBox(object):
    def __init__(self, x0, y0, x1, y1):
        super(BBox, self).__init__()

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.check_order()

    def check_order(self):
        if ( self.x0 > self.x1 ):
            self.x0, self.x1 = self.x1, self.x0
        
        if ( self.y0 > self.y1 ):
            self.y0, self.y1 = self.y1, self.y0

    def get_shape(self):
        return ( self.y1 - self.y0 + 1, self.x1 - self.x0 + 1 )

    def is_inside(self, shape):
        return ( 
            self.x0 >= 0 and self.y0 >= 0 and
            self.x1 < shape[1] and self.y1 < shape[0] )

    def expand(self, shape):
        localShape = self.get_shape()

        x0 = self.x0
        y0 = self.y0
        x1 = self.x1
        y1 = self.y1

        if ( localShape[0] < shape[0] ):
            p0, p1 = compute_padding( localShape[0], shape[0] )
            y0 -= p0
            y1 += p1

        if ( localShape[1] < shape[1] ):
            p0, p1 = compute_padding( localShape[1], shape[1] )
            x0 -= p0
            x1 += p1

        return BBox( x0, y0, x1, y1 )

    def rescale(self, rHW):
        shape = self.get_shape()
        locRHW = shape[0] / shape[1]

        if ( locRHW == rHW ):
            return BBox( self.x0, self.y0, self.x1, self.y1 )

        if ( locRHW < rHW ):
            newH = round( rHW * shape[1] )
            p0, p1 = compute_padding(shape[0], newH)
            return BBox( self.x0, self.y0 - p0, self.x1, self.y1 + p1 )

        if ( locRHW > rHW ):
            newW = round( shape[0] / rHW )
            p0, p1 = compute_padding( shape[1], newW )
            return BBox( self.x0 - p0, self.y0, self.x1 + p1, self.y1 )

    def __repr__(self):
        s = 'BBox (%d, %d), (%d, %d)' % ( 
            self.x0, self.y0, self.x1, self.y1 )

        return s

def get_bbox_from_item(item):
    '''
    Arguments:
    item (xml.etree.ElementTree.Element): XML element.

    Returns:
    A bbox object.
    '''

    bndbox = item.find('bndbox')

    if ( bndbox is None ):
        return None
    
    x0 = int(bndbox.find('xmin').text)
    y0 = int(bndbox.find('ymin').text)
    x1 = int(bndbox.find('xmax').text)
    y1 = int(bndbox.find('ymax').text)

    return BBox( x0, y0, x1, y1 )

def parse_items(fn):
    tree = ET.parse(fn)
    root = tree.getroot()

    path = None
    for p in root.iter('path'):
        path = p.text
        break # Only hanle one path element.

    if ( path is None ):
        return None

    # Find out the case name.
    caseName = find_case_name_from_filename(
        os.path.basename(path))

    if ( caseName is None):
        return None

    items = []
    for obj in root.iter('object'):
        for item in obj.findall('item'):
            name = item.find('name').text
            bbox = get_bbox_from_item(item)
            if ( bbox is None ):
                continue

            items.append( { 
                'case': caseName, 
                'name': name,
                'bbox': bbox } )

    return items

def crop(img, bbox):
    '''Crop an image by a bounding box.
    If the bbox is outside the image, then paddings will be applied.

    Arguments:
    img (NumPy array): Image.
    bbox (BBox): Bounding box.

    Returns:
    A crop.
    '''
    H, W = img.shape[:2]

    px0 = -bbox.x0 if bbox.x0 < 0 else 0
    px1 = bbox.x1 - W if bbox.x1 > W else 0
    py0 = -bbox.y0 if bbox.y0 < 0 else 0
    py1 = bbox.y1 - H if bbox.y1 > H else 0

    if ( px0 != 0 or px1 != 0 or py0 != 0 or py1 != 0 ):
        if (img.ndim == 2):
            newShape = ( py0 + H + py1, px0 + W + px1 )
        elif ( img.ndim == 3 ):
            newShape = ( py0 + H + py1, px0 + W + px1, img.shape[2] )
        else:
            raise Exception('Only supports single channel and 3-channel images. image.shape = {}'.format(img.shape))
        newImg = np.zeros( newShape, dtype=img.dtype )
        newImg[ py0:py0+H, px0:px0+W, ... ] = img
    else:
        newImg = img

    return newImg[ 
        bbox.y0+py0:bbox.y1+py0+1, 
        bbox.x0+px0:bbox.x1+px0+1,
        ... ]

def crop_target(img, bbox, targetShape=(128, 128)):
    '''
    Arguments:
    img (NumPy array): Image.
    bbox (BBox): BBox object.
    targetShape (2-element): (H, W).

    Returns:
    A list of crops.
    '''

    if ( not bbox.is_inside( img.shape[:2] ) ):
        raise Exception('Wrong bbox. Image shape {}, BBox is {}. '.format(
            img.shape, bbox))
    
    bboxShape = bbox.get_shape()
    
    if ( bboxShape[0] <= targetShape[0] and 
         bboxShape[1] <= targetShape[1] ):
        # Crop a targetShape.
        newBBox = bbox.expand( targetShape )
        c = crop(img, newBBox)
    else:
        # Crop a larger one then resize.
        newBBox = bbox.rescale( targetShape[0] / targetShape[1] )
        c = crop(img, newBBox)
        c = cv2.resize( c, ( targetShape[1], targetShape[0] ), interpolation=cv2.INTER_LINEAR )

    return c

def resize_and_binarize(img):
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    return binarize(img, 80)

def crop_item_and_save(d, index, item, imageDict, flagTransform=False ):
    '''
    Arguments:
    d (str): Output directory.
    index (int): Item index.
    item (dict): {'case', 'name', 'bbox'}
    imageDict (dict): Mappings from case name to image filename.
    '''
    assert(index >= 0)
    caseName = item['case']

    if ( caseName not in imageDict.keys() ):
        return
    
    outDir = os.path.join( d, '%06d_%s' % (index, caseName) )
    test_directory(outDir)

    # Homography.
    ht = HomographyTransform()

    for i, fn in enumerate(imageDict[caseName]):
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        if ( i == 0 ):
            img0 = img
            if ( flagTransform ):
                imgRB0 = resize_and_binarize(img0)
        else:
            if ( flagTransform ):
                imgRB1 = resize_and_binarize(img)
                h, m = ht.compute_homography( imgRB0, imgRB1, r=5 )
                img = cv2.warpPerspective( 
                    img, h, (img.shape[1], img.shape[0]), 
                    flags=cv2.INTER_LINEAR )
        
        c = crop_target(img, item['bbox'])

        stem = os.path.splitext(os.path.basename(fn))[0]
        outFn = os.path.join(outDir, '%s_%s.png' % (stem, NAME_MAP[ item['name'] ]))
        print(outFn)

        # cv2.imwrite( outFn, c ) # OpenCV cannot handle the encoding correctly.
        cv2.imencode('.png', c)[1].tofile(outFn)

def pool_crop_item_and_save(args):
    return crop_item_and_save(*args)

def handle_args():
    parser = argparse.ArgumentParser(description='Gather the labeled data. ')

    parser.add_argument('labeldir', type=str, 
        help='The folder contains all the labeled data. ')

    parser.add_argument('imagedir', type=str, 
        help='The directory contains all the images. ')

    parser.add_argument('outdir', type=str, 
        help='The output directory. ')

    parser.add_argument('--label-xml-pattern', type=str, default='outputs/**/*-01.xml', 
        help='The search pattern for the labeled data. ')

    parser.add_argument('--image-pattern', type=str, default='*.jpg', 
        help='The search pattern for the images. ')

    parser.add_argument('--transform', action='store_true', default=False, 
        help='Set this flag to transform the images other than the first one. ')

    parser.add_argument('--np', type=int, default=2, \
        help='The process number. ')

    parser.add_argument('--list-all-types', action='store_true', default=False, 
        help='Set this flag to only show the type strings after reading all the XML files. ')

    parser.add_argument('--debug', action='store_true', default=False, 
        help='Set this flag to enable debug mode. ')

    return parser.parse_args()

def main():
    # Handle the arguments.
    args = handle_args()

    # Prepare the output directory.
    test_directory(args.outdir)

    # Find the files.
    labelFnList = find_files( args.labeldir, args.label_xml_pattern )
    imageFnList = find_files( args.imagedir, args.image_pattern )

    print( '%d label files found' % ( len(labelFnList) ) )
    print( '%d image files found' % ( len(imageFnList) ) )

    imageDict = convert_image_filenames_2_dict(imageFnList)
    print( '%d entries in the converted image dictionary. ' % ( len(imageDict) ) )

    itemList = []
    for fn in labelFnList:
        items = parse_items(fn)
        if ( items is not None ):
            for item in items:
                itemList.append(item)
    
    print('%d items found. ' % (len(itemList)))

    if ( args.list_all_types ):
        typeDict = dict()
        for item in itemList:
            try:
                typeDict[item['name']] += 1
            except KeyError:
                typeDict[item['name']] = 1

        print(typeDict.items())
        return 0
    
    poolArgs = [ [args.outdir, i, item, imageDict, args.transform] for i, item in enumerate(itemList) ]

    if ( args.debug ):
        for i, item in enumerate(itemList):
            crop_item_and_save(args.outdir, i, item, imageDict)
            break
    else:
        with Pool(args.np) as p:
            p.map( pool_crop_item_and_save, poolArgs )

    return 0

if __name__ == '__main__':
    import sys
    print('Hello, %s! ' % ( os.path.basename(__file__) ))
    sys.exit( main() )
