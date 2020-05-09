import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from seaborn import heatmap

"""""
This module contains utilities for evaluation & visualization.
"""""


def show_image(img):
    pil_image = Image.fromarray(img)
    pil_image.show()


def dir_files(img_path, img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)), img_ext)
    return img_names


def num2fix_str(x, d):
    st = '%0*d' % (d, x)
    return st


def accuracy(ys, y, st):
    print(st)
    if y.shape[1] > y.shape[0]:
        y = y.transpose()
        ys = ys.transpose()
    if y.shape[1] > 1:
        d = np.argmax(y, axis=1)
        ds = np.argmax(ys, axis=1)
    else:
        d = y
        ds = ys
    c = confusion_matrix(d, ds)
    acc = accuracy_score(d, ds)
    print('Confusion Matrix:')
    print(c)
    print(f'Accuracy: {acc * 100}%')
    print()
    nm = c.shape[0]
    plt.figure(figsize=(7, 5))
    heatmap(c, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlim(0, nm)
    plt.ylim(nm, 0)
    plt.title('Confusion Matrix ' + st, fontsize=14)
    plt.show()
    return acc
