import numpy as np
import cv2
from skimage import transform as tf
from scipy import signal 
from helpers import *

def getFeatures(img,bbox):
    """
    Description: Identify feature points within bounding box for each object
    Input:
        img: Grayscale input image, (H, W)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame, (F, N, 2)
    Instruction: Please feel free to use cv2.goodFeaturesToTrack() or cv.cornerHarris()
    """
    features = np.zeros((0, 25,2))
    for box in bbox.astype(int):
        cropped_img = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        corner = np.int32(cv2.goodFeaturesToTrack(cropped_img, 25, 0.01, 1)) + np.array([[box[0][0], box[0][1]]])
        corner = np.array([corner[:,0,:]])
        features = np.concatenate((features, corner), axis = 0)
    return features.astype(int)

def estimateAllTranslation(features, img1, img2):
    """
    Description: Get corresponding points for all feature points
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
    """

    new_features = np.zeros((0, 25,2))
    for f in features:
        new_features_temp = np.zeros((0,2))
        for coord in f:

            if coord[0] >= 0 and coord[0] < img1.shape[1] and coord[1] >= 0 and coord[1] < img1.shape[0]:
                new_feature = estimateFeatureTranslation(coord.astype(int), img1, img2)
                new_features_temp= np.vstack((new_features_temp, new_feature))
            else:
                new_features_temp= np.zeros((25,2))
        new_features = np.concatenate((new_features, np.array([new_features_temp])), axis = 0)
    return np.array(new_features)



def estimateFeatureTranslation(coord, img1, img2):
    """
    Description: Get corresponding point for one feature point
    Input:
        feature: Coordinate of feature point in first frame, (2,)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_feature: Coordinate of feature point in second frame, (2,)
    Instruction: Please feel free to use interp2() and getWinBound() from helpers
    """

    dx_sum = 0
    dy_sum = 0
    win_size = 15
    window1 = getWinBound(img1.shape, coord[0], coord[1], win_size)
    window2 = getWinBound(img2.shape, coord[0], coord[1], win_size)
    img1_win = img1[int(window1[2]):int(window1[3]), int(window1[0]):int(window1[1])]
    img2_win = img2[int(window2[2]):int(window2[3]), int(window2[0]):int(window2[1])]
    for i in range(20):

        dx, dy = optical_flow(img1_win, img2_win)
        dx_sum += dx
        dy_sum += dy
        if dx > 100:
            breakpoint()
        img2_win = get_new_img(img2, dx_sum, dy_sum, coord, win_size)
    new_feature = np.array([[coord[0]+dx_sum, coord[1]+dy_sum]])
    return new_feature



def select_win(lst, slice1, slice2):
    return [item[slice1, slice2] for item in lst]

def applyGeometricTransformation(features, new_features, bbox, frame):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame after eliminating outliers, (F, N1, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """
    new_features_transform = np.zeros((0, 25,2))

    threshold = 1
    stop_track = {}
    for index, feature in enumerate(features):
        if bbox.shape[0] != 0 and bbox.shape[1] != 0:
            tform = tf.estimate_transform('similarity', feature, new_features[index])
            features_tf = tform(feature) 

            bbox[index] = tform(bbox[index])
            features_corrected = np.zeros((0, features_tf.shape[1]))
        


            for idx, feature_point in enumerate(feature): #optimize--- since they're vectors, just subtract regularly!
                if np.hypot(*(features_tf[idx] - feature_point)) <= threshold:
                    if features_tf[idx][0] >= bbox[index][0][0] and features_tf[idx][0] <= bbox[index][1][0] \
                    and features_tf[idx][1] >= bbox[index][0][1] and features_tf[idx][1] <= bbox[index][1][1]:
                        features_corrected = np.vstack((features_corrected, features_tf[idx]))
            
                
            if features_corrected.shape[0] < 25:
                more_features, cutoff_signal = getMoreFeatures(frame, bbox[index].astype(int), 25-features_corrected.shape[0])
                if not cutoff_signal:
                    features_corrected = np.vstack((features_corrected, more_features))
                    new_features_transform = np.concatenate((new_features_transform, np.array([features_corrected])), axis = 0)
                else:
                    stop_track[index] = True
                    new_features_transform = np.concatenate((new_features_transform, np.zeros((1,25,2))), axis = 0)
            else:
                new_features_transform = np.concatenate((new_features_transform, np.array([features_corrected])), axis = 0)

            
    return new_features_transform.astype(int), bbox, stop_track


def get_new_img(img, dx, dy, coord, win_size):
    img2_win = getWinBound(img.shape, coord[0], coord[1], win_size)
    x, y = np.meshgrid(np.arange(int(img2_win[0]), int(img2_win[1])), np.arange(int(img2_win[2]), int(img2_win[3])))
    new_x, new_y = x + dx, y + dy
    return interp2(img, new_x, new_y)

def optical_flow(img1, img2):
    Ix, Iy = findGradient(img2) #don't compute gradient each time!!
    It = img2 - img1
    A = np.hstack((Ix.reshape(-1, 1), Iy.reshape(-1, 1)))
    b = -It.reshape(-1, 1)
    res = np.linalg.solve(A.T @ A, A.T @ b)
    return res[0, 0], res[1, 0]

def findGradient(img, ksize=10, sigma=1):
    G = cv2.getGaussianKernel(ksize, sigma)
    G = G @ G.T
    fx = np.array([[1, -1]])
    fy = fx.T
    Gx = signal.convolve2d(G, fx, 'same', 'symm')[:, 1:]
    Gy = signal.convolve2d(G, fy, 'same', 'symm')[1:, :]
    Ix = signal.convolve2d(img, Gx, 'same', 'symm')
    Iy = signal.convolve2d(img, Gy, 'same', 'symm')
    return Ix, Iy
    

def getMoreFeatures(img,box, num_features):
    cropped_img = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    if cropped_img.shape[0] != 0 and cropped_img.shape[1] != 0:


        corner = np.int32(cv2.goodFeaturesToTrack(cropped_img, num_features, 0.01, 1)) + np.array([[box[0][0],box[0][1]]])
        features = corner[:,0,:]
        cutoff_signal = False
    else:
        features = np.array([])
        cutoff_signal = True
    return features.astype(int), cutoff_signal
