import numpy as np
from src.deep_learning.perceptron.activation import Linear
from src.deep_learning.perceptron.loss import MSELoss
from src.deep_learning.perceptron import Perceptron


def main():
    n_features = 3
    n_examples = 5
    X = np.random.randint(low=-5, high=5, size=(n_examples, n_features)).tolist()
    W_ = np.random.randint(low=-10, high=10, size=n_features + 1).tolist()

    # Perceptron being made just to generate training labels
    gen = Perceptron(n_features=n_features)
    gen.weights = W_
    y = []
    for x in X:
        y.append(gen.forward(X=x))

    neuron = Perceptron(n_features=n_features, activation=Linear, loss=MSELoss)
    n_epochs = 1000
    print(f"Loss = {neuron.get_loss(X=X, y=y)}")
    for _ in range(n_epochs):
        y_pred = [neuron.forward(X=x) for x in X]
        dJ_da = neuron.loss.backward(y, y_pred)
        neuron.backward(dJ_da=dJ_da)
        print(f"Loss at step {_ + 1} = {neuron.loss.forward(y, y_pred)}")


if __name__ == "__main__":
    main()
