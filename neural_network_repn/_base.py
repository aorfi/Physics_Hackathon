"""Utilities for the neural network modules
"""

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# License: BSD 3 clause

import numpy as np

from scipy.special import expit as logistic_sigmoid


def identity(X):
    """Simply return the input array.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Same as the input data.
    """
    return X


def logistic(X):
    """Compute the logistic function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return logistic_sigmoid(X, out=X)


def tanh(X):
    """Compute the hyperbolic tan function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return np.tanh(X, out=X)


def relu(X):
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X

def log(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    temp = (X==0)
    X[temp] = 1
    return np.log(abs(X))

def repn2(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**2

def repn3(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**3

def repn4(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**4

def repn5(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**5

def repn6(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**6

def repn7(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**7

def repn8(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**8

def repn9(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**9

def repn10(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**10

def repn11(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**11

def repn12(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**12

def repn13(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**13

def repn14(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**14

def repn15(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**15

def repn16(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**16

def repn17(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**17

def repn18(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**18

def repn19(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**19

def repn20(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**20

def repn21(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**21

def repn22(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**22

def repn23(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**23

def repn24(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**24

def repn25(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**25

def repn26(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**26

def repn27(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**27

def repn28(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**28

def repn29(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**29

def repn30(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X**30

def softmax(X):
    """Compute the K-way softmax function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'logistic': logistic,
               'relu': relu, 'softmax': softmax, 'log': log,
               'repn2': repn2, 'repn3': repn3,
               'repn4': repn4, 'repn5': repn5, 
               'repn6': repn6, 'repn7': repn7, 
               'repn8': repn8, 'repn9': repn9,
               'repn10': repn10, 'repn11': repn11,
               'repn12': repn12, 'repn13': repn13,
               'repn14': repn14, 'repn15': repn15, 
               'repn16': repn16, 'repn17': repn17, 
               'repn18': repn18, 'repn19': repn19,
               'repn20': repn20, 'repn21': repn21,
               'repn22': repn22, 'repn23': repn23,
               'repn24': repn24, 'repn25': repn25, 
               'repn26': repn26, 'repn27': repn27, 
               'repn28': repn28, 'repn29': repn29, 'repn30': repn30}

               
def inplace_identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do


def inplace_logistic_derivative(Z, delta):
    """Apply the derivative of the logistic sigmoid function.

    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= (1 - Z)


def inplace_tanh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic tanh function.

    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= (1 - Z ** 2)


def inplace_relu_derivative(Z, delta):
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z == 0] = 0

def inplace_log_derivative(Z, delta):
    temp = (Z==0)
    Z[temp] = 1
    delta /= Z**2
    delta[temp] = 0

def inplace_repn2_derivative(Z, delta):
    num=2
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn3_derivative(Z, delta):
    num=3
    # delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn4_derivative(Z, delta):
    num=4
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn5_derivative(Z, delta):
    num=5
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn6_derivative(Z, delta):
    num=6
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn7_derivative(Z, delta):
    num=7
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn8_derivative(Z, delta):
    num=8
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn9_derivative(Z, delta):
    num=9
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn10_derivative(Z, delta):
    num=10
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn11_derivative(Z, delta):
    num=11
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn12_derivative(Z, delta):
    num=12
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn13_derivative(Z, delta):
    num=13
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn14_derivative(Z, delta):
    num=14
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn15_derivative(Z, delta):
    num=15
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn16_derivative(Z, delta):
    num=16
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn17_derivative(Z, delta):
    num=17
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn18_derivative(Z, delta):
    num=18
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn19_derivative(Z, delta):
    num=19
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn20_derivative(Z, delta):
    num=20
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn21_derivative(Z, delta):
    num=21
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn22_derivative(Z, delta):
    num=22
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn23_derivative(Z, delta):
    num=23
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn24_derivative(Z, delta):
    num=24
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn25_derivative(Z, delta):
    num=25
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn26_derivative(Z, delta):
    num=26
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn27_derivative(Z, delta):
    num=27
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn28_derivative(Z, delta):
    num=28
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn29_derivative(Z, delta):
    num=29
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0

def inplace_repn30_derivative(Z, delta):
    num=30
    delta *= num*np.power(Z,(num-1)/num)
    delta[Z == 0] = 0



DERIVATIVES = {'identity': inplace_identity_derivative,
               'tanh': inplace_tanh_derivative,
               'logistic': inplace_logistic_derivative,
               'relu': inplace_relu_derivative,
               'log': inplace_log_derivative,
               'repn2': inplace_repn2_derivative,
               'repn3': inplace_repn3_derivative,
               'repn4': inplace_repn4_derivative,
               'repn5': inplace_repn5_derivative,
               'repn6': inplace_repn6_derivative,
               'repn7': inplace_repn7_derivative,
               'repn8': inplace_repn8_derivative,
               'repn9': inplace_repn9_derivative,
               'repn10': inplace_repn10_derivative,
               'repn11': inplace_repn11_derivative,
               'repn12': inplace_repn12_derivative,
               'repn13': inplace_repn13_derivative,
               'repn14': inplace_repn14_derivative,
               'repn15': inplace_repn15_derivative,
               'repn16': inplace_repn16_derivative,
               'repn17': inplace_repn17_derivative,
               'repn18': inplace_repn18_derivative,
               'repn19': inplace_repn19_derivative,
               'repn20': inplace_repn20_derivative,
               'repn21': inplace_repn21_derivative,
               'repn22': inplace_repn22_derivative,
               'repn23': inplace_repn23_derivative,
               'repn24': inplace_repn24_derivative,
               'repn25': inplace_repn25_derivative,
               'repn26': inplace_repn26_derivative,
               'repn27': inplace_repn27_derivative,
               'repn28': inplace_repn28_derivative,
               'repn29': inplace_repn29_derivative,
               'repn30': inplace_repn30_derivative}

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return ((y_true - y_pred) ** 2).mean() / 2


def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true * np.log(y_prob)) / y_prob.shape[0]


def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.

    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    return -np.sum(y_true * np.log(y_prob) +
                   (1 - y_true) * np.log(1 - y_prob)) / y_prob.shape[0]


LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss}
