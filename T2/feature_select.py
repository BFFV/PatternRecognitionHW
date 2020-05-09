from pybalu.feature_selection import sfs as s

"""""
This module contains functions for feature selection.
"""""


def sfs(features, classes, n_features=20):
    return s(features, classes, n_features, method="fisher")
