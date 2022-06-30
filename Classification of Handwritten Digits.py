STAGE 1

import keras.datasets.mnist
import tensorflow as tf
import numpy as np

tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def main():
    unique_classes = np.unique(y_train)
    flatten_2D_x_train = np.array([x_train[i].flatten() for i in range(len(x_train))])
    feature_shape = flatten_2D_x_train.shape
    target_shape = y_train.shape
    feature_array_max = flatten_2D_x_train.max()
    feature_array_min = flatten_2D_x_train.min()

    print(f"Classes: {unique_classes}")
    print(f"Features' shape: {feature_shape}")
    print(f"Targets' shape: {target_shape}")
    print(f"min: {feature_array_min}, max: {feature_array_max}")


if __name__ == '__main__':
    main()
