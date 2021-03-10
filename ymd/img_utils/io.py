
import cv2
import os

from CommonPython.Filesystem import Filesystem

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '
    return cv2.imread(fn, cv2.IMREAD_UNCHANGED)

def write_image(fn, img):
    Filesystem.test_directory_by_filename(fn)
    cv2.imwrite(fn, img)
