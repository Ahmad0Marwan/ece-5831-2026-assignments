import numpy as np
from multilayer_perceptron import MultilayerPerceptron


def main():
    np.set_printoptions(precision=6, suppress=True)
    mlp = MultilayerPerceptron()
    examples = [
        np.array([1.0, 0.5, -1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 1.0, 0.2])
    ]

    for i, x in enumerate(examples, start=1):
        a1, z1, a2, z2, a3, z3, a4, y = mlp.forward(
            x, return_layers=True
        )
        print(f"Example {i}")
        print("Input:", x)
        print("Hidden layer 1 (sigmoid):", z1)
        print("Hidden layer 2 (sigmoid):", z2)
        print("Hidden layer 3 (sigmoid):", z3)
        print("Output before ReLU:", a4)
        print("Output after ReLU:", y)
        print()


if __name__ == "__main__":
    main()
