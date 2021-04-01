# coding: utf-8

# Author: Yaoyu Hu <yyhu_live@outlook.com>

import cv2
import numpy as np
import time

from . import export

@export
class HomographyTransform(object):
    def __init__(self, hessian=1000):
        self.detector = cv2.xfeatures2d_SURF.create(hessianThreshold=hessian)
        
        indexParams  = { "algorithm": 1, "trees":5 }
        searchParams = { "checks": 50 }
        self.flann   = cv2.FlannBasedMatcher( indexParams, searchParams )
    
    def compute_homography(self, refImg, tstImg, r=None):
        # Feature extraction.
        kpQ, descQ = self.detector.detectAndCompute(refImg, mask=None)
        kpD, descD = self.detector.detectAndCompute(tstImg, mask=None)

        # Matching.
        matches = self.flann.knnMatch( descD, descQ, k=2 )
        goodMatches = [ m for m, n in matches if m.distance < 0.7 * n.distance ]

        # Keypoints.
        srcKP = np.array( [ kpD[ m.queryIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        dstKP = np.array( [ kpQ[ m.trainIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )

        # Ratio.
        if ( r is not None ):
            srcKP *= r
            dstKP *= r

        # Homography.
        H, mask = cv2.findHomography( srcKP, dstKP, cv2.RANSAC, 5.0 )

        return H, mask

    def transform(self, refImg, tstImg, toBeTransformed=None, r=None):
        H, mask = self.compute_homography( refImg, tstImg, r )
        if ( toBeTransformed is None ):
            return cv2.warpPerspective( tstImg, H, (tstImg.shape[1], tstImg.shape[0]), flags=cv2.INTER_LINEAR )
        else:
            return cv2.warpPerspective( toBeTransformed, H, (toBeTransformed.shape[1], toBeTransformed.shape[0]), flags=cv2.INTER_LINEAR )

@export
class HomographyCpu(object):
    def __init__(self, hessian=1000):
        super(HomographyCpu, self).__init__()
        self.detector = cv2.xfeatures2d_SURF.create(hessianThreshold=hessian)
        self.matcher  = cv2.BFMatcher(cv2.NORM_L2)
    
    def __call__(self, refImg, tstImg):
        timeStart = time.time()
        # Feature extraction.
        kpQ, descQ = self.detector.detectAndCompute(refImg, mask=None)
        kpD, descD = self.detector.detectAndCompute(tstImg, mask=None)

        # Matching.
        matches = self.matcher.knnMatch( descD, descQ, k=2 )
        goodMatches = [ m for m, n in matches if m.distance < 0.7 * n.distance ]
        timeDetectionAndMatching = time.time()

        # Homography.
        srcKP = np.array( [ kpD[ m.queryIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        dstKP = np.array( [ kpQ[ m.trainIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        H, mask = cv2.findHomography( srcKP, dstKP, cv2.RANSAC, 3.0 )
        timeHomography = time.time()
        
        return H, goodMatches, timeDetectionAndMatching - timeStart, timeHomography - timeDetectionAndMatching

    @staticmethod
    def scale_homography_matrix( hMat, curShape, oriShape ):
        # Two scale factors.
        fx = oriShape[1] / curShape[1]
        fy = oriShape[0] / curShape[0]

        # The scale matrices.
        s  = np.eye( 3, dtype=np.float32 )
        si = s.copy() # Invert of s.

        s[0, 0] = fx
        s[1, 1] = fy

        si[0, 0] = 1. / fx
        si[1, 1] = 1. / fy

        return s @ hMat @ si