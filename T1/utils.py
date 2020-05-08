import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

"""""
This module contains utilities for images/features visualization.
"""""


# Displays an image in OpenCV
def show_img(img):
    out = cv2.resize(img, (1280, 720))
    cv2.imshow('img.png', out)
    cv2.waitKey()


# Visualization of the classification results
def show_classes(img, classes):
    # Map component labels
    label_numbers = np.uint8(img)
    blank_ch = 255 * np.ones_like(label_numbers)
    classified_img = cv2.merge([label_numbers, blank_ch, blank_ch])

    # RGB display
    classified_img = cv2.cvtColor(classified_img, cv2.COLOR_HSV2BGR)

    # Set background & classes
    colors = {1: (0, 0, 255), 2: (0, 255, 255), 3: (255, 0, 0),
              4: (128, 0, 128), 5: (0, 255, 0)}
    classified_img[label_numbers != 0] = 0
    for c in classes:
        classified_img[label_numbers == c] = colors[classes[c]['Class']]
    classified_img[label_numbers == 0] = 0
    out = cv2.resize(classified_img, (1280, 720))
    cv2.imshow('classes.png', out)
    cv2.waitKey()


# Visualization of each letter detected in the image
def show_components(labels, highlight=None, duration=None):
    # Map component labels
    label_numbers = np.uint8(labels)
    blank_ch = 255*np.ones_like(label_numbers)
    labeled = cv2.merge([label_numbers, blank_ch, blank_ch])

    # RGB display
    labeled = cv2.cvtColor(labeled, cv2.COLOR_HSV2BGR)

    # Set background
    labeled[label_numbers == 0] = 0

    # Can highlight a specific letter
    if highlight:
        labeled[label_numbers == highlight] = 255
    out = cv2.resize(labeled, (1280, 720))
    cv2.imshow('components.png', out)
    if duration:
        cv2.waitKey(duration)
    else:
        cv2.waitKey()


# Visualize each letter detected & it's contour
def check_components(labeled, contours, num_components, start=0, duration=None):
    for num in range(start, num_components):
        show_components(labeled, highlight=num, duration=duration)
        show_components(contours, highlight=num, duration=duration)


# 1D Histogram
def show_histogram(class_data, features, feature):
    data = {}
    for c in range(1, 6):
        data[c] = [comp['Features'][features.index(feature)]
                   for comp in class_data if comp['Class'] == c]
    bins = np.linspace(-2, 3, 15)
    classes = {1: 'A', 2: 'S', 3: 'D', 4: 'F', 5: 'G'}
    plt.hist([data[c] for c in classes], bins, alpha=0.5,
             label=classes.values())
    plt.legend(loc='upper right')
    plt.show()


# Visualization of the confusion matrix
def show_matrix(data, classes):
    df_cm = pd.DataFrame(data, classes, classes)
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.show()
