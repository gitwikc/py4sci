import numpy as np

from src.deep_learning.perceptron.models import Layer


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: float) -> float:
    return np.max(0, x)


def d_relu(x: float) -> float:
    return np.max(0, x) / x


Sigmoid = Layer(name="sigmoid", forward=sigmoid, backward=d_sigmoid)
ReLU = Layer(name="relu", forward=relu, backward=d_relu)
Linear = Layer(name="linear", forward=lambda x: x, backward=lambda x: 1)
