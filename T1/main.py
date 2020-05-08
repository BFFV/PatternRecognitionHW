import numpy as np
import char_classifier as cc
from collections import deque
from utils import show_classes


# Training classes & images
LETTERS = {1: 'A', 2: 'S', 3: 'D', 4: 'F', 5: 'G'}
TRAINING_1 = list(18 * '1') + list(18 * '2') + list(18 * '3') + \
                   list(17 * '4') + list(2 * '5') + list(1 * '4') + \
                   list(16 * '5') + list(17 * '1') + list(2 * '2') + \
                   list(1 * '1') + list(13 * '2') + list(2 * '3') + \
                   list(2 * '2') + list(1 * '3') + list(1 * '2') + \
                   list(9 * '3') + list(3 * '4') + list(2 * '3') + \
                   list(3 * '4') + list(3 * '3') + list(4 * '4') + \
                   list(1 * '3') + list(7 * '4') + list(3 * '5') + \
                   list(1 * '4') + list(15 * '5')
TRAINING_2 = list(18 * '1') + list(18 * '2') + list(18 * '3') + \
             list(18 * '4') + list(18 * '5') + list(18 * '1') + \
             list(18 * '2') + list(18 * '3') + list(18 * '4') + list(18 * '5')
TRAINING_CLASSES = [int(c) for c in TRAINING_1] + [int(c) for c in TRAINING_2]
TRAINING_IMAGES = ['Training_01.png', 'Training_02.png']
TESTING = list(18 * '1') + list(18 * '2') + list(18 * '3') + \
          list(18 * '4') + list(18 * '5') + list(18 * '1') + \
          list(18 * '2') + list(18 * '3') + list(18 * '4') + list(18 * '5')


# Trains the classifier with a list of images
def training(images, classes, features):
    _id = 0
    amount = len(images)
    data = deque()
    for image in images:
        print(f'Training {amount * "."}')
        # Extract important features for all letters in the image
        img_data, labeled_img = cc.extract_data(image, features)
        # Store data
        for letter in img_data:
            data.append(
                {'Features': np.array(
                    [img_data[letter][f] for f in img_data[letter] if f in
                     features]), 'Class': classes[_id]})
            _id += 1
        amount -= 1
        # show_components(labeled_img)
    print('Finished Training!')
    return data


# Applies the k-NN classifier to an image
def test_img(img_path, train_data, features, norm_method, k=5):
    print('Testing ..')
    # Extract important features for all letters in the image
    img_data, labeled_img = cc.extract_data(img_path, features)
    print(f'Detected {len(img_data)} Characters In Image')
    # Normalization
    print('Testing .')
    cc.normalize(train_data, img_data, features, norm_method)
    # Classify each letter
    for letter in img_data:
        img_data[letter]['Class'] = cc.knn(
            k, np.array([img_data[letter][f] for f in img_data[letter] if f in
                         features]), train_data)
    print('Finished Testing!')
    # Calculate Accuracy
    correct_predictions = sum(
        [1 for x in zip([str(img_data[letter]['Class']) for letter in img_data],
                        TESTING) if x[0] == x[1]])
    accuracy = correct_predictions / len(TESTING)
    print(f'Total Accuracy: {round(accuracy * 100, 2)}%')
    # Visualize the results in the original image
    print('Classification:\n\nA: RED\nS: YELLOW\nD: BLUE\nF: PURPLE\nG: GREEN')
    show_classes(labeled_img, img_data)


selected_features = ['Roundness', 'Hu_1', 'Hu_3', 'Hu_4', 'Hu_7']
training_data = training(TRAINING_IMAGES, TRAINING_CLASSES, selected_features)
test_img('Testing.png', training_data, selected_features, 'z_score', k=1)
"""""
for f in selected_features:
    show_histogram(training_data, selected_features, f)
"""""
"""""
results = []
for i in range(len(LETTERS)):
    results.append([int(x) for x in list(i * '0')] + [18] +
                   [int(x) for x in list((len(LETTERS) - i - 1) * '0')])
print(results)
show_matrix(results, LETTERS.values())
"""""
