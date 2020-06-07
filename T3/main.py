import numpy as np
from pybalu.feature_selection import clean
from pybalu.feature_transformation import normalize
from feature_extract import extract_features
from feature_select_transform import select_features, transform_features
from classification import classify
from utils import accuracy


# Train or load from data file

load = False
data_file_0 = 'X0_train.npy'  # Normal
data_file_1 = 'X1_train.npy'  # Pneumonia
data_file_2 = 'X2_train.npy'  # COVID19

# Image Classification Strategy

# Features to extract (features)
selected_features = ['lbp']

# Selection/Transformation steps
processing_strategy = [['sfs', {'n_features': 50, 'method': 'fisher'}]]

# Classifier to use
classifier = ['knn', {'n_neighbors': 3}]

# Training: Feature Extraction
print('Training...')
if load:  # Load extracted features from data file
    X0_train = np.load(data_file_0)
    print('Training..')
    X1_train = np.load(data_file_1)
    print('Training.')
    X2_train = np.load(data_file_2)
else:  # Extract features from all images & save the data
    X0_train = extract_features('data/train/class_0', 'png', selected_features)
    np.save('X0_train.npy', X0_train)
    print('Training..')
    X1_train = extract_features('data/train/class_1', 'png', selected_features)
    np.save('X1_train.npy', X1_train)
    print('Training.')
    X2_train = extract_features('data/train/class_2', 'png', selected_features)
    np.save('X2_train.npy', X2_train)

# Training: Data Set
print('Training Subset:')
X_train = np.concatenate((X0_train, X1_train, X2_train), axis=0)
d0_train = np.zeros([X0_train.shape[0], 1], dtype=int)
d1_train = np.ones([X1_train.shape[0], 1], dtype=int)
d2_train = np.full([X2_train.shape[0], 1], 2, dtype=int)
d_train = np.concatenate((d0_train, d1_train, d2_train), axis=0)
print(f'Original Extracted Features: {X_train.shape[1]} ({X_train.shape[0]} '
      f'samples)')

# Training: Cleaning
print('Cleaning...')
s_clean = clean(X_train, show=True)
X_train_clean = X_train[:, s_clean]
print(f'           cleaned features: {X_train_clean.shape[1]} '
      f'({X_train_clean.shape[0]} samples)')

# Training: Normalization
print('Normalizing...')
X_train_norm, a, b = normalize(X_train_clean)
print(f'        normalized features: {X_train_norm.shape[1]} '
      f'({X_train_norm.shape[0]} samples)')

# Training: Processing Strategy
print('Selecting/Transforming Features...')
X_train_final = X_train_norm
for index, step in enumerate(processing_strategy):
    if step[0] in ['sfs']:
        step[1]['n_features'] = \
            min(X_train_final.shape[1], step[1]['n_features'])
        output = select_features(X_train_final, d_train, step[0], step[1])
        X_train_final = X_train_final[:, output]
        processing_strategy[index].append(output)
    else:
        output = transform_features(step[0], step[1])
        X_train_final = output[0]
        processing_strategy[index].append(output[1])
print(f'          selected/transformed features: {X_train_final.shape[1]} '
      f'({X_train_final.shape[0]} samples)')

# Testing: Feature Extraction
print('Testing...')
X0_test = extract_features('data/test/class_0', 'png', selected_features)
print('Testing..')
X1_test = extract_features('data/test/class_1', 'png', selected_features)
print('Testing.')
X2_test = extract_features('data/test/class_2', 'png', selected_features)

# Testing: Data Set
print('Testing Subset:')
X_test = np.concatenate((X0_test, X1_test, X2_test), axis=0)
d0_test_a = np.zeros([X0_test.shape[0], 1], dtype=int)
d0_test_b = np.zeros([X0_test.shape[0] // 10, 1], dtype=int)
d1_test_a = np.ones([X1_test.shape[0], 1], dtype=int)
d1_test_b = np.ones([X1_test.shape[0] // 10, 1], dtype=int)
d2_test_a = np.full([X2_test.shape[0], 1], 2, dtype=int)
d2_test_b = np.full([X2_test.shape[0] // 10, 1], 2, dtype=int)
d_test_a = np.concatenate((d0_test_a, d1_test_a, d2_test_a), axis=0)
d_test_b = np.concatenate((d0_test_b, d1_test_b, d2_test_b), axis=0)

# Testing: Cleaning
print('Cleaning...')
X_test_clean = X_test[:, s_clean]

# Testing: Normalization
print('Normalizing...')
X_test_norm = X_test_clean * a + b

# Testing: Processing Strategy
print('Selecting/Transforming Features...')
X_test_final = X_test_norm
for index, step in enumerate(processing_strategy):
    if step[0] in ['sfs']:
        selected = step[2]
        X_test_final = X_test_final[:, selected]
    elif step[0] == 'pca':
        params = step[2]
        X_test_final = np.matmul(X_test_final - params['Xm'], params['A'])
print(f'    clean+norm+selected/transformed features: {X_test_final.shape[1]} '
      f'({X_test_final.shape[0]} samples)')

# Classification
print('Classifying...\n')
results_a, results_b = classify(
    X_train_final, X_test_final, d_train, classifier[0], classifier[1])
accuracy(results_a, d_test_a, 'Patch Classification')
accuracy(results_b, d_test_b, 'Patient Classification')
print('Finished!')
