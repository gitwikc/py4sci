from typing import List

import numpy as np

from src.deep_learning.perceptron.models import Loss


def binary_cross_entropy_loss(y_true: List[float], y_pred: List[float]) -> float:
    return np.linalg.norm(
        [-y * np.log(p) - (1 - y) * np.log(1 - p) for y, p in zip(y_true, y_pred)]
    )


def mse_loss(y_true: List[float], y_pred: List[float]) -> float:
    return np.mean([[(yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)]])


def loss_backward(y_true: List[float], y_pred: List[float]) -> float:
    return np.mean([y1 - y2 for y1, y2 in zip(y_pred, y_true)])


BinaryCrossEntropy = Loss(
    name="binary_cross_entropy",
    forward=binary_cross_entropy_loss,
    backward=loss_backward,
)

MSELoss = Loss(name="mse_loss", forward=mse_loss, backward=loss_backward)
