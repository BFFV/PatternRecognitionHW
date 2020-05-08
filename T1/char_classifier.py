import cv2
import math
import numpy as np
from collections import deque

"""""
This module contains functions for feature extraction & classification.
"""""


# Finds the boundary lines for each letter & draws them in a new image
def find_edges(segmented):
    height, width = segmented.shape
    output_img = cv2.drawContours(
        np.zeros((height, width)), cv2.findContours(
            segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, 255, 1)
    return np.uint8(output_img)


# Calculates the area for each letter
def find_area(components):
    labels, counts = np.unique(components, return_counts=True)
    areas_dict = dict(zip(labels, counts))
    # The background is discarded
    del areas_dict[0]
    # Really small areas are discarded as noise
    # (pieces of a letter that are barely separated from it)
    return {label: area for label, area in areas_dict.items() if area > 500}


# Calculates the perimeter for each letter
def find_perimeter(contours, components):
    labels, counts = np.unique(contours, return_counts=True)
    perimeters_dict = dict(zip(labels, counts))
    return {label: perimeters_dict[label] for label in components}


# Finds the coordinate of every pixel for each letter
def find_pixels(components):
    pixels_dict = {}
    height, width = components.shape
    for row in range(height):
        for px in range(width):
            if components[row][px]:
                if components[row][px] not in pixels_dict:
                    pixels_dict[components[row][px]] = deque()
                pixels_dict[components[row][px]].append((px, row))
    return pixels_dict


# Calculates the central moment of a letter, given r & s
def central_moment(r, s, centroid, pixels):
    return sum([((px[0] - centroid[0]) ** r) * ((px[1] - centroid[1]) ** s)
                for px in pixels])


# Calculates the eta moment of a letter, given r & s (Hu-Moments)
def eta_moment(r, s, centroid, pixels):
    t = (r + s / 2) + 1
    return central_moment(r, s, centroid, pixels) / (len(pixels) ** t)


# Calculates the Hu-Moments of a letter
def hu_moments(centroid, pixels, number):
    hu_m = 0
    if number == 1:
        hu_m = eta_moment(2, 0, centroid, pixels) + \
               eta_moment(0, 2, centroid, pixels)
    elif number == 2:
        hu_m = ((eta_moment(2, 0, centroid, pixels) -
                eta_moment(0, 2, centroid, pixels)) ** 2) + 4 * (
                eta_moment(1, 1, centroid, pixels) ** 2)
    elif number == 3:
        hu_m = ((eta_moment(3, 0, centroid, pixels) -
                 3 * eta_moment(1, 2, centroid, pixels)) ** 2) + \
               ((3 * eta_moment(2, 1, centroid, pixels) -
                 eta_moment(0, 3, centroid, pixels)) ** 2)
    elif number == 4:
        hu_m = ((eta_moment(3, 0, centroid, pixels) +
                 eta_moment(1, 2, centroid, pixels)) ** 2) + \
               ((eta_moment(2, 1, centroid, pixels) +
                 eta_moment(0, 3, centroid, pixels)) ** 2)
    elif number == 5:
        hu_m = (eta_moment(3, 0, centroid, pixels) -
                3 * eta_moment(1, 2, centroid, pixels)) * \
               (eta_moment(3, 0, centroid, pixels) +
                eta_moment(1, 2, centroid, pixels)) * (
                       ((eta_moment(3, 0, centroid, pixels) +
                         eta_moment(1, 2, centroid, pixels)) ** 2) - 3 *
                       ((eta_moment(2, 1, centroid, pixels) +
                         eta_moment(0, 3, centroid, pixels)) ** 2)) \
               + (3 * eta_moment(2, 1, centroid, pixels) -
                  eta_moment(0, 3, centroid, pixels)) * (
                       eta_moment(2, 1, centroid, pixels) +
                       eta_moment(0, 3, centroid, pixels)) * (
                       3 * ((eta_moment(3, 0, centroid, pixels) +
                             eta_moment(1, 2, centroid, pixels)) ** 2) -
                       ((eta_moment(2, 1, centroid, pixels) +
                         eta_moment(0, 3, centroid, pixels)) ** 2))
    elif number == 6:
        hu_m = (eta_moment(2, 0, centroid, pixels) -
                eta_moment(0, 2, centroid, pixels)) * (
                       ((eta_moment(3, 0, centroid, pixels) +
                         eta_moment(1, 2, centroid, pixels)) ** 2) -
                       ((eta_moment(2, 1, centroid, pixels) +
                         eta_moment(0, 3, centroid, pixels)) ** 2)) \
               + 4 * eta_moment(1, 1, centroid, pixels) * \
               (eta_moment(3, 0, centroid, pixels) +
                eta_moment(1, 2, centroid, pixels)) * (
                       eta_moment(2, 1, centroid, pixels) +
                       eta_moment(0, 3, centroid, pixels))
    elif number == 7:
        hu_m = (3 * eta_moment(2, 1, centroid, pixels) -
                eta_moment(0, 3, centroid, pixels)) * \
               (eta_moment(3, 0, centroid, pixels) +
                eta_moment(1, 2, centroid, pixels)) * (
                       ((eta_moment(3, 0, centroid, pixels) +
                         eta_moment(1, 2, centroid, pixels)) ** 2) - 3 *
                       ((eta_moment(2, 1, centroid, pixels) +
                         eta_moment(0, 3, centroid, pixels)) ** 2)) \
               - (eta_moment(3, 0, centroid, pixels) -
                  3 * eta_moment(1, 2, centroid, pixels)) * (
                       eta_moment(2, 1, centroid, pixels) +
                       eta_moment(0, 3, centroid, pixels)) * (
                       3 * ((eta_moment(3, 0, centroid, pixels) +
                             eta_moment(1, 2, centroid, pixels)) ** 2) -
                       ((eta_moment(2, 1, centroid, pixels) +
                         eta_moment(0, 3, centroid, pixels)) ** 2))
    return hu_m


# Normalizes the data
def normalize(train_data, test_data, features, method):
    for i, f in enumerate(features):
        current_feature = [comp['Features'][i] for comp in train_data]
        if method == 'range':
            norm_data = (min(current_feature), max(current_feature))
        else:
            avg = sum(current_feature) / len(current_feature)
            s_dev = math.sqrt(sum([(d - avg) ** 2 for d in current_feature]) /
                              (len(current_feature) - 1))
            norm_data = (avg, s_dev)
        for index, comp in enumerate(train_data):
            if method == 'range':
                train_data[index]['Features'][i] = \
                    (comp['Features'][i] - norm_data[0]) / \
                    (norm_data[1] - norm_data[0])
            else:
                train_data[index]['Features'][i] = \
                    (comp['Features'][i] - norm_data[0]) / norm_data[1]
        for letter in test_data:
            if method == 'range':
                test_data[letter][f] = \
                    (test_data[letter][f] - norm_data[0]) / \
                    (norm_data[1] - norm_data[0])
            else:
                test_data[letter][f] = (test_data[letter][f] - norm_data[0]) / \
                                       norm_data[1]


# Extracts selected features from each letter in an image
def extract_data(img, features):
    # Read the image
    image = cv2.imread(img, 0)

    # Image segmentation for binary images
    segmented_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

    # Draw image with edges (boundaries of each letter)
    edged_img = find_edges(segmented_img)

    # Find each letter in the segmented/edged image
    num_labels, labeled_img = cv2.connectedComponents(segmented_img)
    num_edges, contour_img = cv2.connectedComponents(edged_img)

    # Used for checking if each letter & it's contour are correct
    # check_components(labeled_img, contour_img, num_edges, duration=200)

    # Calculate the Area for each letter, discarding isolated pixels
    comp_areas = find_area(labeled_img)

    # Calculate the Perimeter for each letter previously detected
    comp_perimeters = find_perimeter(contour_img, comp_areas)

    # Get the pixels for each letter
    comp_pixels = find_pixels(labeled_img)

    # Basic data for each letter
    comp_dict = {
        comp: {'Area': comp_areas[comp], 'Perimeter': comp_perimeters[comp],
               'Pixels': comp_pixels[comp]} for comp in comp_areas}

    # Calculate Centroids
    for comp in comp_dict:
        current = comp_dict[comp]
        centroid_x = sum([x[0] for x in current['Pixels']]) / current['Area']
        centroid_y = sum([x[1] for x in current['Pixels']]) / current['Area']
        current['Centroid'] = (round(centroid_x, 2), round(centroid_y, 2))

    # Calculate Roundness
    if 'Roundness' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Roundness'] = round((4 * current['Area'] * math.pi) / (
                    current['Perimeter'] ** 2), 3)

    # Calculate Hu-Moments
    if 'Hu_1' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Hu_1'] = hu_moments(
                current['Centroid'], current['Pixels'], 1)
    if 'Hu_2' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Hu_2'] = hu_moments(
                current['Centroid'], current['Pixels'], 2)
    if 'Hu_3' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Hu_3'] = hu_moments(
                current['Centroid'], current['Pixels'], 3)
    if 'Hu_4' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Hu_4'] = hu_moments(
                current['Centroid'], current['Pixels'], 4)
    if 'Hu_5' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Hu_5'] = hu_moments(
                current['Centroid'], current['Pixels'], 5)
    if 'Hu_6' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Hu_6'] = hu_moments(
                current['Centroid'], current['Pixels'], 6)
    if 'Hu_7' in features:
        for comp in comp_dict:
            current = comp_dict[comp]
            current['Hu_7'] = hu_moments(
                current['Centroid'], current['Pixels'], 7)

    return comp_dict, labeled_img


# Classifies a letter using the k-NN algorithm
def knn(k, letter, train_data):
    ordered = sorted(train_data, key=lambda x: np.linalg.norm(
        letter - x['Features']))
    neighbors = [n['Class'] for n in ordered[:k]]
    chosen = max(set(neighbors), key=neighbors.count)
    return chosen
