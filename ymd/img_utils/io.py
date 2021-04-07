
import cv2
import os

from CommonPython.Filesystem import Filesystem

def read_image(fn):
    if ( not os.path.isfile(fn) ):
        raise Exception( f'{fn} does not exist. ' )
    
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

    if ( img is None ):
        raise Exception( f'Cannot read {fn}. File might be corrupted. ')

    return img

def write_image(fn, img):
    Filesystem.test_directory_by_filename(fn)
    cv2.imwrite(fn, img)
