from typing import List, Tuple
import numpy as np
from src.deep_learning.perceptron.loss import BinaryCrossEntropy
from src.deep_learning.perceptron.activation import Linear
from src.deep_learning.perceptron.models import Layer, Loss


class Perceptron:
    def __init__(
        self,
        n_features: int = 1,
        activation: Layer = Linear,
        loss: Loss = BinaryCrossEntropy,
        alpha: float = 0.003,
    ) -> None:
        self.weights = np.random.randn(n_features + 1)
        self.activation = activation
        self.loss = loss
        self.alpha = alpha
        self.a = self.z = self.da_dz = self.X = 0

    def get_weights(self) -> Tuple[List[float], float]:
        *weights, bias = self.weights
        return weights, bias

    def get_loss(self, X: List[List[float]], y: List[float]) -> float:
        y_pred = [self.forward(x) for x in X]
        return self.loss.backward(y, y_pred)

    def forward(self, X: List[float]) -> float:
        self.X = X + [1.0]
        self.z = np.sum([w * x for w, x in zip(self.weights, self.X)])
        self.a = self.activation.forward(self.z)
        self.da_dz = self.activation.backward(self.z)
        return self.a

    def backward(self, dJ_da: List[float]) -> None:
        dJ_dz = dJ_da * self.da_dz
        dJ_dW = [X * dJ_dz for X in self.X]
        self.weights = [W - self.alpha * dJ_dw for W, dJ_dw in zip(self.weights, dJ_dW)]
