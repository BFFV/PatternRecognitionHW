import numpy as np
from sklearn.neighbors import KNeighborsClassifier

"""""
This module contains functions for image classification.
"""""


# Classifies images
def classify(train_data, test_data, classes, classifier, kwargs):
    patch_classes = []
    if classifier == 'knn':
        knn = KNeighborsClassifier(**kwargs)
        knn.fit(train_data, classes)
        patch_classes = knn.predict(test_data)
    groups = np.array_split(patch_classes, patch_classes.shape[0] / 10)
    patient_classes = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 1, groups)
    return patch_classes, patient_classes
