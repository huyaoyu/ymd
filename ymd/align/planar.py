# coding: utf-8

# Author: Yaoyu Hu <yyhu_live@outlook.com>

import cv2
import numpy as np
import time

from . import export

def pad_homogeneous_coordinate(x):
    '''x must be DxN, where D is the dimension of the point 
    and N is the number of points.'''
    hc = np.ones( x.shape[1], dtype=x.dtype ).reshape( (1, -1) )
    return np.concatenate( (x, hc), axis=0 )

@export
class HomographyTransform(object):
    def __init__(self, hessian=1000):
        self.detector = cv2.xfeatures2d_SURF.create(hessianThreshold=hessian)
        
        indexParams  = { "algorithm": 1, "trees":5 }
        searchParams = { "checks": 50 }
        self.flann   = cv2.FlannBasedMatcher( indexParams, searchParams )
    
    def compute_homography(self, refImg, tstImg, r=None):
        # Feature extraction.
        kpDst, descDst = self.detector.detectAndCompute(refImg, mask=None)
        kpSrc, descSrc = self.detector.detectAndCompute(tstImg, mask=None)

        # Matching.
        matches = self.flann.knnMatch( descSrc, descDst, k=2 )
        goodMatches = [ m for m, n in matches if m.distance < 0.7 * n.distance ]

        # Keypoints.
        srcKP = np.array( [ kpSrc[ m.queryIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        dstKP = np.array( [ kpDst[ m.trainIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )

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
        kpDst, descDst = self.detector.detectAndCompute(refImg, mask=None)
        kpSrc, descSrc = self.detector.detectAndCompute(tstImg, mask=None)

        # Matching.
        matches = self.matcher.knnMatch( descSrc, descDst, k=2 )
        goodMatches = [ m for m, n in matches if m.distance < 0.7 * n.distance ]
        timeDetectionAndMatching = time.time()

        # Homography.
        srcKP = np.array( [ kpSrc[ m.queryIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        dstKP = np.array( [ kpDst[ m.trainIdx ].pt for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        H, mask = cv2.findHomography( srcKP, dstKP, cv2.RANSAC, 3.0 )
        timeHomography = time.time()

        # Compute the error of the projected keypoints.
        nHomographyMatched = mask.sum()

        if ( H is not None ):
            mask  = mask.reshape((-1,)).astype(np.bool)
            proj  = cv2.perspectiveTransform(srcKP[mask, ...], H)
            diffV = dstKP[mask, ...] - proj
            diff  = np.linalg.norm( diffV.reshape((-1, 2)), axis=1 ).mean()
            retFlag = True
        else:
            diff = -1
            retFlag = False

        return retFlag, H, goodMatches, nHomographyMatched, diff, timeDetectionAndMatching - timeStart, timeHomography - timeDetectionAndMatching

    @staticmethod
    def scale_homography_matrix( hMat, curDstShape, oriDstShape, curSrcShape, oriSrcShape ):
        # Two scale factors for the souce image.
        fx = curSrcShape[1] / oriSrcShape[1]
        fy = curSrcShape[0] / oriSrcShape[0]

        # The scale matrix.
        fSrc = np.eye( 3, dtype=np.float32 )
        fSrc[0, 0] = fx
        fSrc[1, 1] = fy

        # Two scale factors for the destination image.
        fx = curDstShape[1] / oriDstShape[1]
        fy = curDstShape[0] / oriDstShape[0]

        # The inverse scale matix
        fDstInv = np.eye( 3, dtype=np.float32 )
        fDstInv[0, 0] = 1. / fx
        fDstInv[1, 1] = 1. / fy

        return fDstInv @ hMat @ fSrc