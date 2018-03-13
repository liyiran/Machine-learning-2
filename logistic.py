from __future__ import division, print_function

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression

    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    w = np.append(b, w)
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    X = np.matrix(X)
    w = np.matrix(w)
    y = np.matrix(y)
    w = binary_train_v2(X, y, w, max_iterations, step_size, N)
    w = np.array(w)
    b = w[0][0]
    w = w[0][1:]
    assert w.shape == (D,)
    return w, b


def binary_train_v2(x, y, w, max_iteration, learning_rate, N):
    for k in range(max_iteration):
        w, converge = gradient_descent_v2(x, y, w, learning_rate, N)
        if converge:
            break
    return w


def gradient_descent_v2(x, y, w, learning_rate, number):
    linear_value = np.sum(np.multiply(w, x), axis=1)
    pred = sigmoid(linear_value)
    error = pred - y.transpose()
    error_matrix = np.multiply(error, x)
    gradient = np.multiply(learning_rate, np.sum(error_matrix, axis=0))
    if gradient.nonzero():
        w -= gradient / number
        return w, False
    else:
        return w, True


def binary_predict_adapter(x, w):
    # W and X has been augment
    linear_value = np.sum(np.multiply(w, x), axis=1)
    return np.rint(sigmoid(linear_value)).astype(int).transpose()


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N)

    # expand w
    w = np.append(b, w)
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    w = np.matrix(w)
    X = np.matrix(X)
    preds = binary_predict_adapter(X, w)
    preds = np.squeeze(np.asarray(preds.transpose()))
    assert preds.shape == (N,)
    return preds


def multinomial_train(X, y, C,
                      w0=None,
                      b0=None,
                      step_size=0.5,
                      max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement a multinomial logistic regression for multiclass 
    classification. Keep in mind, that for this task you may need a 
    special (one-hot) representation of classification labels, where 
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to. 
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    w = np.matrix(np.c_[b, w])
    X = np.matrix(np.append(np.ones((X.shape[0], 1)), X, axis=1))
    y = generate_y(y, N, C)
    w = multi_train_stochastic(X, y, w, max_iterations, step_size, N, C, D, softmax)
    b = w[:, 0]
    b = np.array(b).reshape((C,))
    w = np.array(np.delete(w, 0, 1))
    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    return e / np.sum(e)  # ndim = 2


def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N)

    w = np.c_[b, w]
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    w = w.reshape(C, 1, D + 1)
    X = np.tile(X, (C, 1, 1))
    linear_value = np.sum(np.multiply(w, X), axis=2, keepdims=True)
    pred = softmax(linear_value)
    preds = np.argmax(pred, axis=0).reshape((N,))
    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using binary classifier and 
    one-versus-rest strategy. Recall, that the OVR classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    w = np.matrix(np.c_[b, w])
    X = np.matrix(np.append(np.ones((X.shape[0], 1)), X, axis=1))
    y = generate_y(y, N, C)
    w = multi_train_v2(X, y, w, max_iterations, step_size, N, C, D, sigmoid)
    b = w[:, 0]
    b = np.array(b).reshape((C,))
    w = np.array(np.delete(w, 0, 1))
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def multi_train_v2(x, y, w, max_iteration, learning_rate, N, C, D, prob_fun):
    for k in range(max_iteration):
        linear_value = np.einsum('CD,ND->CN', w, x)
        pred = prob_fun(linear_value)
        error = pred - y
        gradient = np.einsum('CN,ND->CD', error, x) * learning_rate
        if gradient.nonzero():
            w -= gradient / N
        else:
            return w
    return w


def multi_train_stochastic(x, y, w, max_iteration, learning_rate, N, C, D, prob_fun):
    for k in range(max_iteration):
        for n in range(x.shape[0]):  # n rows
            x_n = x[n:n + 1, :]
            y_n = y[:, n:n + 1]
            linear_value = np.einsum('CD,ND->CN', w, x_n)
            pred = prob_fun(linear_value)
            error = pred - y_n
            gradient = np.multiply(learning_rate, np.einsum('CN,ND->CD', error, x_n))
            w -= gradient
    return w


def generate_y(y, N, C):
    big_y = np.tile(y, (C, 1))
    x = 0
    new_y = np.empty((0))
    for r in big_y:
        new_y = np.append(new_y, (r == x).astype(int))
        x += 1
    return np.matrix(new_y.reshape(big_y.shape))


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and predictions from binary
    classifier. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N)

    w = np.c_[b, w]
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    w = w.reshape(C, 1, D + 1)
    X = np.tile(X, (C, 1, 1))
    linear_value = np.sum(np.multiply(w, X), axis=2, keepdims=True)
    pred = sigmoid(linear_value)
    preds = np.argmax(pred, axis=0).reshape((N,))
    assert preds.shape == (N,)
    return preds


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)


def run_binary():
    from data_loader import toy_data_binary, \
        data_loader_mnist

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()

    w, b = binary_train(X_train, y_train)

    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' %
          (accuracy_score(y_train, train_preds),
           accuracy_score(y_test, preds)))

    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train]
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test]

    w, b = binary_train(X_train, binarized_y_train)

    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' %
          (accuracy_score(binarized_y_train, train_preds),
           accuracy_score(binarized_y_test, preds)))


def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
        toy_data_multiclass_5_classes, \
        data_loader_mnist

    datasets = [(toy_data_multiclass_3_classes_non_separable(),
                 'Synthetic data', 3),
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5),
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data

        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' %
              (accuracy_score(y_train, train_preds),
               accuracy_score(y_test, preds)))

        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' %
              (accuracy_score(y_train, train_preds),
               accuracy_score(y_test, preds)))


if __name__ == '__main__':

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
