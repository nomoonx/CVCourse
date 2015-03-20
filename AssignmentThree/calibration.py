import numpy as np
import cv2
import glob
import os
import fnmatch

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points. It's a chessboard, so z should be zero.
# Since the width of side is 30mm, the points should be (0,0),(30,0),(60,0)..
pattern_size = (8, 5)
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= 30

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# case sensitivity
images = glob.glob('*.JPG')
images.extend(glob.glob('*.jpg'))

for fname in images:
    print fname,
    print '    ..........     ',
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # image is too big to display, so resize it
        cv2.drawChessboardCorners(img, (8, 5), corners, ret)
        img_height, img_width = img.shape[:2]
        scale = 0.3
        resized_image = cv2.resize(img, (int(scale * img_width), int(scale * img_height)))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', int(scale * img_width), int(scale * img_height))
        cv2.imshow('img', resized_image)
        cv2.waitKey(500)

        print 'OK'

    else:
        print 'can not find the corners'

# calculate calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print 'matrix'
print mtx
print 'distortion coefficients'
print dist
cv2.destroyAllWindows()
