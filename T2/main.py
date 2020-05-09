import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pybalu.feature_selection import clean
from pybalu.feature_transformation import normalize
from feature_extract import extract_features
from feature_select import sfs
from utils import accuracy


# Training: Feature Extraction
print('Training...')
X0_train = extract_features('Training_0', 'png')  # Edit
print('Training..')
X1_train = extract_features('Training_1', 'png')  # Edit

# Training: Data Set
print('Training Subset:')
X_train = np.concatenate((X0_train, X1_train), axis=0)
d0_train = np.zeros([X0_train.shape[0], 1], dtype=int)
d1_train = np.ones([X1_train.shape[0], 1], dtype=int)
d_train = np.concatenate((d0_train, d1_train), axis=0)
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

# Training: Feature Selection
print('Selecting Features...')
s_sfs = sfs(X_train_norm, d_train, n_features=50)
X_train_sfs = X_train_norm[:, s_sfs]
print(f'          selected features: {X_train_sfs.shape[1]} '
      f'({X_train_sfs.shape[0]} samples)')

# Testing: Feature Extraction
print('Testing...')
X0_test = extract_features('Testing_0', 'png')  # Edit
print('Testing..')
X1_test = extract_features('Testing_1', 'png')  # Edit

# Testing: Data Set
print('Testing Subset:')
X_test = np.concatenate((X0_test, X1_test), axis=0)
d0_test = np.zeros([X0_test.shape[0], 1], dtype=int)
d1_test = np.ones([X1_test.shape[0], 1], dtype=int)
d_test = np.concatenate((d0_test, d1_test), axis=0)

# Testing: Cleaning
print('Cleaning...')
X_test_clean = X_test[:, s_clean]

# Testing: Normalization
print('Normalizing...')
X_test_norm = X_test_clean * a + b

# Testing: Feature Selection
print('Selecting Features...')
X_test_sfs = X_test_norm[:, s_sfs]
print(f'    clean+norm+selected features: {X_test_sfs.shape[1]} '
      f'({X_test_sfs.shape[0]} samples)')

# Classification
print('Classifying...')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_sfs, d_train)
ds = knn.predict(X_test_sfs)
accuracy(ds, d_test, 'Finished!')
