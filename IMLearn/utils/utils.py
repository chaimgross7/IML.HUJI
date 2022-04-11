from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """

    # n = X.shape[0]
    #
    # # randomly shuffle the rows
    # permute = np.random.permutation(n)
    # X = X.iloc[permute]
    # y = y.iloc[permute]
    #
    # split_index = int(np.ceil(n * train_proportion))
    #
    # train_X, train_y, test_X, test_y = X[:split_index], y[:split_index], X[split_index:], y[split_index:]
    #
    # return train_X, train_y, test_X, test_y
    df = pd.concat([X, y], axis=1)
    n = df.shape[0]
    frac = int(np.ceil(0.75 * X.shape[0]))
    train = df.sample(frac)
    test = df[~df.index.isin(train.index)]
    train_X, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
    test_X, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
    return train_X, train_y, test_X, test_y

def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
