import numpy as np

"""""
This module contains functions for feature selection.
"""""


# Calculates the Fisher Score for separability
def fisher_score(features, classification):
    # There is an equal amount of samples for each class
    p = np.array([[0.5], [0.5]])  # Proportion of each class (equal)
    n_features = features.shape[1]  # Amount of features to evaluate

    # Centroid of all samples (Z)
    class_centroid = features.mean(0)

    # Within class covariance
    cov_within = np.zeros(shape=(n_features, n_features))

    # Between class covariance
    cov_between = np.zeros(shape=(n_features, n_features))

    # Calculate the covariance matrix
    norm = classification.ravel() - classification.min()
    last_class = norm.max() + 1
    for c in range(last_class):
        indexes = (norm == c)  # Class k indexes
        class_features = features[indexes, :]  # Class k samples
        class_mean = class_features.mean(0)  # Class k centroid
        class_cov = np.cov(class_features, rowvar=False)  # Class k covariance
        cov_within += p[c] * class_cov  # Within class covariance
        # Between class covariance
        difference = (class_mean - class_centroid).reshape((n_features, 1))
        cov_between += p[c] * difference @ difference.T
    return np.trace(np.linalg.inv(cov_within) @ cov_between)  # Spur


# Sequential Forward Selection
def sfs(features, classes, n_features=50):
    selected_features = []
    i, f = features.shape
    remaining_features = set(np.arange(f))
    current_data = np.zeros((i, 0))  # Samples for the selected features

    # Gets the score for the current selected features
    def get_score(current):
        feats = np.hstack([current_data, features[:, current].reshape(-1, 1)])
        return fisher_score(feats, classes)

    # Select the best 'n_features' features
    for n in range(n_features):
        best = max(remaining_features, key=get_score)
        selected_features.append(best)
        remaining_features.remove(best)
        current_data = np.hstack(
            [current_data, features[:, best].reshape(-1, 1)])
    return np.array(selected_features)
