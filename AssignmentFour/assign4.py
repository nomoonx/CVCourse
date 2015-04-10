__author__ = 'noMoon'
import cv2
import cv
import numpy as np

pts1 = []
pts2 = []

# load camera matrix and distortion matrix from xml
left_intr_file = 'Calibration-Parameters/intr_L.xml'
left_camera_matrix = np.asarray(cv.Load(left_intr_file, cv.CreateMemStorage(), 'CameraMatrix'))
left_distortion = np.asarray(cv.Load(left_intr_file, cv.CreateMemStorage(), 'Distortion'))

right_intr_file = 'Calibration-Parameters/intr_R.xml'
right_camera_matrix = np.asarray(cv.Load(right_intr_file, cv.CreateMemStorage(), 'CameraMatrix'))
right_distortion = np.asarray(cv.Load(right_intr_file, cv.CreateMemStorage(), 'Distortion'))

# detect 128 pairs pictures
for j in range(1, 129):
    left_image_name = 'Images/Left/L_' + str(j) + '.bmp'
    right_image_name = 'Images/Right/R_' + str(j) + '.bmp'
    img1 = cv2.imread(left_image_name, 0)  # left image
    img2 = cv2.imread(right_image_name, 0)  # right image
    # undistort image
    dist1 = cv2.undistort(img1, left_camera_matrix, left_distortion)
    dist2 = cv2.undistort(img2, right_camera_matrix, right_distortion)

    # find the keypoints and descriptors with SUFT
    surf = cv2.SURF(400)
    kp1, des1 = surf.detectAndCompute(dist1, None)
    kp2, des2 = surf.detectAndCompute(dist2, None)

    bfMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bfMatcher.match(des1, des2)
    for m in matches:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float64(pts1)
pts2 = np.float64(pts2)
# use 8-point algorithm to calculate the Fundamental Matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
print 'Fundamental Matrix from pictures'
print F
print '-------------------------------------------------'

# calculate Fundamental Matrix from parameters
ext_L2R_file = 'Calibration-Parameters/ext_L_To_R.xml'
R = np.asarray(cv.Load(ext_L2R_file, cv.CreateMemStorage(), 'R'))
T = np.asarray(cv.Load(ext_L2R_file, cv.CreateMemStorage(), 'T'))
T = np.transpose(T)
S = np.array([[0, -T[0, 2], T[0, 1]], [T[0, 2], 0, -T[0, 0]], [-T[0, 1], T[0, 0], 0]])
E = R.dot(S)
FM = np.transpose(np.linalg.inv(right_camera_matrix)).dot(E).dot(np.linalg.inv(left_camera_matrix))
print 'Fundamental Matrix from L2R parameters'
print FM
print '-------------------------------------------------'

ext_R2L_file = 'Calibration-Parameters/ext_R_To_L.xml'
R = np.asarray(cv.Load(ext_R2L_file, cv.CreateMemStorage(), 'R'))
T = np.asarray(cv.Load(ext_R2L_file, cv.CreateMemStorage(), 'T'))
T = np.transpose(T)
S = np.array([[0, -T[0, 2], T[0, 1]], [T[0, 2], 0, -T[0, 0]], [-T[0, 1], T[0, 0], 0]])
E = R.dot(S)
FM = np.transpose(np.linalg.inv(right_camera_matrix)).dot(E).dot(np.linalg.inv(left_camera_matrix))
print 'Fundamental Matrix from R2L parameters'
print FM
