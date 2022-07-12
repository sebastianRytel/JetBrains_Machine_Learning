====================== STAGE 1 ======================

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

====================== STAGE 2 ======================

import keras.datasets.mnist
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train_full_data, y_train_full_data), (x_test_full_data, y_test_full_data) = keras.datasets.mnist.load_data()


def main():
    x_train_2d = np.array([x_train_full_data[i].flatten() for i in range(len(x_train_full_data))])

    x_train_samples = x_train_2d[:6000]
    y_train_samples = y_train_full_data[:6000]

    X_train_samples, X_test_samples, y_train_samples, y_test_samples = train_test_split(x_train_samples,
                                                                                        y_train_samples,
                                                                                        test_size=0.3, random_state=40)

    samples_proportion = pd.Series(y_train_samples).value_counts(normalize=True)

    print(f"x_train shape: {np.array(X_train_samples).shape}")
    print(f"x_test shape: {np.array(X_test_samples).shape}")
    print(f"y_train shape: {np.array(y_train_samples).shape}")
    print(f"y_test shape: {np.array(y_test_samples).shape}")
    print("Proportion of samples per class in train set:")
    print(samples_proportion.round(2))


if __name__ == '__main__':
    main()

====================== STAGE 3 ======================
    
import keras.datasets.mnist
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score

tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train_full_data, y_train_full_data), (x_test_full_data, y_test_full_data) = keras.datasets.mnist.load_data()


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    pred = model.predict(features_test)
    score = precision_score(pred, target_test, average='macro')
    print(f'Model: {model}\nAccuracy: {score}\n')
    return score

def main():
    x_train_2d = np.array([x_train_full_data[i].flatten() for i in range(len(x_train_full_data))])

    x_train_samples = x_train_2d[:6000]
    y_train_samples = y_train_full_data[:6000]

    X_train_samples, X_test_samples, y_train_samples, y_test_samples = train_test_split(x_train_samples,
                                                                                        y_train_samples,
                                                                                        test_size=0.3, random_state=40)

    k_nearest = KNeighborsClassifier()
    logistic_regr = LogisticRegression()
    random_forest = RandomForestClassifier(random_state=40)
    decision_tree = DecisionTreeClassifier(random_state=40)

    score_k_nearest = fit_predict_eval(model=k_nearest,
                     features_train=X_train_samples,
                     features_test=X_test_samples,
                     target_train=y_train_samples,
                     target_test=y_test_samples)

    score_decision_tree = fit_predict_eval(model=decision_tree,
                        features_train=X_train_samples,
                        features_test=X_test_samples,
                        target_train=y_train_samples,
                        target_test=y_test_samples)

    score_logistic_regr = fit_predict_eval(model=logistic_regr,
                        features_train=X_train_samples,
                        features_test=X_test_samples,
                        target_train=y_train_samples,
                        target_test=y_test_samples)


    score_random_forest = fit_predict_eval(model=random_forest,
                        features_train=X_train_samples,
                        features_test=X_test_samples,
                        target_train=y_train_samples,
                        target_test=y_test_samples)


    dict_of_scores = {"KNeighborsClassifier": score_k_nearest,
                      "DecisionTreeClassifier": score_decision_tree,
                      "LogisticRegression": score_logistic_regr,
                      "RandomForestClassifier": score_random_forest,
                      }
    best_classifier = max(dict_of_scores)

    print(f"The answer to the question: {best_classifier} - {dict_of_scores[best_classifier]}")

if __name__ == '__main__':
    main()
