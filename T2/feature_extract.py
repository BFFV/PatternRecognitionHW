import numpy as np
import cv2
from pybalu.feature_extraction import lbp_features
from utils import show_image, dir_files

"""""
This module contains functions for feature extraction.
"""""


def get_image(path, show=False):
    img = cv2.imread(path)
    if show:
        show_image(img)
    return img


def extract_features_img(st):
    img = get_image(st)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_0 = lbp_features(gray, hdiv=1, vdiv=1, mapping='nri_uniform')
    x_red = lbp_features(img[:, :, 0], hdiv=1, vdiv=1, mapping='nri_uniform')
    x_green = lbp_features(img[:, :, 1], hdiv=1, vdiv=1, mapping='nri_uniform')
    x_blue = lbp_features(img[:, :, 2], hdiv=1, vdiv=1, mapping='nri_uniform')
    x_lbp = np.concatenate((x_0, x_red, x_green, x_blue))
    features = np.asarray(x_lbp)
    return features


def extract_features(dir_path, fmt):
    st = '*.' + fmt
    img_names = dir_files(dir_path + '/', st)
    n = len(img_names)
    data = []
    for i in range(n):
        img_path = img_names[i]
        features = extract_features_img(dir_path + '/' + img_path)
        if i == 0:
            m = features.shape[0]
            data = np.zeros((n, m))
            # print('size of extracted features:')
            # print(features.shape)
        data[i] = features
    return data
