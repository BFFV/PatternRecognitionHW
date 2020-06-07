import numpy as np
import cv2
from skimage.feature import hog
from pybalu.feature_extraction import lbp_features, haralick_features, \
    gabor_features
from utils import show_image, dir_files

"""""
This module contains functions for feature extraction.
"""""


# Reads an image
def get_image(path, show=False):
    img = cv2.imread(path)
    if show:
        show_image(img)
    return img


# Extracts features from an image
def extract_features_img(image, selected):
    img = get_image(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray scale image
    features = np.array([])
    if 'lbp' in selected:  # Local Binary Patterns
        lbp = lbp_features(img, hdiv=1, vdiv=1, mapping='nri_uniform')
        features = np.concatenate((features, lbp))
    if 'hog' in selected:  # Histogram of Gradients (Gray + RGB)
        hog_features = hog(img, orientations=16, pixels_per_cell=(64, 64),
                           cells_per_block=(1, 1))
        features = np.concatenate((features, hog_features))
    if 'haralick' in selected:  # Haralick Textures (Gray + RGB)
        haralick = haralick_features(img, distance=3)
        features = np.concatenate((features, haralick))
    if 'gabor' in selected:  # Gabor Features
        gabor = gabor_features(img, rotations=4, dilations=4)
        features = np.concatenate((features, gabor))
    return features


# Extracts features for all images in dir_path
def extract_features(dir_path, fmt, selected):
    st = '*.' + fmt
    img_names = dir_files(dir_path + '/', st)
    n = len(img_names)
    data = []
    for i in range(n):
        img_path = img_names[i]
        features = extract_features_img(dir_path + '/' + img_path, selected)
        if not i:
            m = features.shape[0]
            data = np.zeros((n, m))
        data[i] = features
    return data
