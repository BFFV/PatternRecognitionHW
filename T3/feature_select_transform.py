import numpy as np
from pybalu.feature_selection import sfs

"""""
This module contains functions for feature selection/transformation.
"""""


# Selects features
def select_features(train_data, classes, operation, kwargs):
    if operation == 'sfs':  # Sequential Forward Selection
        return sfs(train_data, classes, **kwargs)


# Transforms features
def transform_features(image, selected):
    if 'lbp' in selected:  # Local Binary Patterns (Gray + RGB)
        features = []
    return
