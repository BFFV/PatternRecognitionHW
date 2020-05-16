import numpy as np
import cv2
from pybalu.feature_extraction import lbp_features, hog_features, \
    haralick_features
from utils import show_image, dir_files

"""""
This module contains functions for feature extraction.
"""""


def get_image(path, show=False):
    img = cv2.imread(path)
    if show:
        show_image(img)
    return img


# Extracts features from an image
def extract_features_img(image, selected):
    img = get_image(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray scale image
    features = np.array([])
    if 'lbp' in selected:  # Local Binary Patterns (Gray + RGB)
        lbp_gray = lbp_features(gray, hdiv=1, vdiv=1, mapping='nri_uniform')
        lbp_red = lbp_features(
            img[:, :, 0], hdiv=1, vdiv=1, mapping='nri_uniform')
        lbp_green = lbp_features(
            img[:, :, 1], hdiv=1, vdiv=1, mapping='nri_uniform')
        lbp_blue = lbp_features(
            img[:, :, 2], hdiv=1, vdiv=1, mapping='nri_uniform')
        lbp = np.concatenate((lbp_gray, lbp_red, lbp_green, lbp_blue))
        features = np.concatenate((features, lbp))
    if 'hog' in selected:  # Histogram of Gradients
        hog = hog_features(gray, v_windows=1, h_windows=1, n_bins=8)
        features = np.concatenate((features, hog))
    if 'haralick' in selected:  # Haralick Textures (Gray + RGB)
        haralick_gray = haralick_features(gray, distance=2)
        haralick_red = haralick_features(img[:, :, 0], distance=2)
        haralick_green = haralick_features(img[:, :, 1], distance=2)
        haralick_blue = haralick_features(img[:, :, 2], distance=2)
        haralick = np.concatenate(
            (haralick_gray, haralick_red, haralick_green, haralick_blue))
        features = np.concatenate((features, haralick))
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
