"""
two_layer_net.py
----------------
A two-layer (one hidden layer) fully-connected neural network whose
gradients are computed with *numerical differentiation* (central
differences), following "Deep Learning from Scratch", Chapter 4.

Architecture:  input -> Affine -> sigmoid -> Affine -> softmax
Loss:          cross-entropy error on one-hot (or integer) labels
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation / loss helpers
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    # subtract the row-wise max for numerical stability
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    """y: network output (N, C) ; t: one-hot (N, C) or integer labels (N,)"""
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    # if labels are one-hot, convert to class indices
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_gradient(f, x):
    """Central-difference numerical gradient of scalar function f at array x."""
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]

        x[idx] = tmp + h
        fxh1 = f(x)          # f(x + h)

        x[idx] = tmp - h
        fxh2 = f(x)          # f(x - h)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp         # restore
        it.iternext()

    return grad


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    # forward pass ---------------------------------------------------------
    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # loss -----------------------------------------------------------------
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # accuracy -------------------------------------------------------------
    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    # numerical gradient ---------------------------------------------------
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for key in ("W1", "b1", "W2", "b2"):
            grads[key] = numerical_gradient(loss_W, self.params[key])
        return grads

    # one SGD step ---------------------------------------------------------
    def update(self, x, t, learning_rate):
        grads = self.numerical_gradient(x, t)
        for key in self.params:
            self.params[key] -= learning_rate * grads[key]
