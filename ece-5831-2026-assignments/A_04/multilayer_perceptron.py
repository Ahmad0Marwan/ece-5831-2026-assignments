import numpy as np


class MultilayerPerceptron:
    """A 3 -> 5 -> 5 -> 2 -> 2 network using NumPy."""

    def __init__(self):
        # Each weight matrix has shape (previous layer, next layer).
        self.network = {}
        self.network["w1"] = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, -0.1, 0.4, -0.3, 0.1],
            [-0.3, 0.4, 0.1, 0.2, -0.2]
        ])
        self.network["b1"] = np.array([0.1, -0.1, 0.2, 0.0, 0.1])

        self.network["w2"] = np.array([
            [0.2, -0.1, 0.3, 0.1, 0.4],
            [0.1, 0.3, -0.2, 0.4, 0.2],
            [-0.3, 0.2, 0.1, -0.1, 0.3],
            [0.4, 0.1, 0.2, 0.3, -0.2],
            [0.2, -0.3, 0.4, 0.2, 0.1]
        ])
        self.network["b2"] = np.array([0.0, 0.1, -0.1, 0.2, 0.0])

        self.network["w3"] = np.array([
            [0.3, -0.2],
            [0.1, 0.4],
            [-0.2, 0.3],
            [0.4, 0.1],
            [0.2, -0.3]
        ])
        self.network["b3"] = np.array([0.1, -0.1])

        self.network["w4"] = np.array([
            [0.5, -0.4],
            [0.3, 0.2]
        ])
        self.network["b4"] = np.array([0.1, -0.2])

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def relu(self, s):
        return np.maximum(0, s)

    def forward(self, x, return_layers=False):
        x = np.asarray(x, dtype=float)
        if x.shape != (3,):
            raise ValueError("Input must contain exactly 3 values.")

        # Hidden layer 1: 3 inputs -> 5 neurons, sigmoid.
        a1 = np.dot(x, self.network["w1"]) + self.network["b1"]
        z1 = self.sigmoid(a1)

        # Hidden layer 2: 5 inputs -> 5 neurons, sigmoid.
        a2 = np.dot(z1, self.network["w2"]) + self.network["b2"]
        z2 = self.sigmoid(a2)

        # Hidden layer 3: 5 inputs -> 2 neurons, sigmoid.
        a3 = np.dot(z2, self.network["w3"]) + self.network["b3"]
        z3 = self.sigmoid(a3)

        # Output layer: 2 inputs -> 2 neurons, ReLU.
        a4 = np.dot(z3, self.network["w4"]) + self.network["b4"]
        y = self.relu(a4)

        if return_layers:
            return a1, z1, a2, z2, a3, z3, a4, y
        return y
