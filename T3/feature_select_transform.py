import numpy as np
from pybalu.feature_selection import sfs
from pybalu.feature_transformation import pca
from sklearn.feature_selection import mutual_info_classif

"""""
This module contains functions for feature selection/transformation.
"""""


# Selects features
def select_features(train_data, classes, operation, kwargs):
    if operation == 'sfs':  # Sequential Forward Selection
        return sfs(train_data, classes, **kwargs)
    elif operation == 'mutual_info':  # Mutual Information
        m_info = mutual_info_classif(
            train_data, classes, n_neighbors=kwargs['n_neighbors'])
        return np.argpartition(
            m_info, -kwargs['n_features'])[-kwargs['n_features']:]


# Transforms features
def transform_features(train_data, operation, kwargs):
    if operation == 'pca':  # Principal Component Analysis
        features, _, a, xm, _ = pca(train_data, **kwargs)
        return features, {'A': a, 'Xm': xm}
