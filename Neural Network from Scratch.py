============== STAGE 1 ==============

import math

import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt

def scale(X_train, X_test):
    X_train_rescaled = np.array(list(map(lambda x: x / max(x), [i for i in X_train])))
    X_test_rescaled = np.array(list(map(lambda x: x / max(x), [i for i in X_test])))
    return list(np.concatenate(([X_train_rescaled[2, 778]], [X_test_rescaled[0, 774]])))


def xavier(n_in, n_out):
    xavier_func = math.sqrt(6) / math.sqrt(n_in + n_out)
    flattened = (np.random.uniform(-xavier_func, xavier_func, (n_in, n_out))).flatten()
    return list(flattened)

def sigmoid(array):
    return list(map(lambda x: 1 / (1 + math.pow(math.e, -x)), array))


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    X_train_rescaled = list(map(lambda x: x / max(x), [i for i in X_train]))

    print(scale(X_train, X_test))
    print(xavier(2, 3))

    print(sigmoid([-1, 0, 1, 2]))
