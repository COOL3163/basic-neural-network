# test accuracy of ANN using MNIST test dataset
import numpy as np
import pickle
from train import NeuralNetwork, load_data, one_hot_encode, softmax, relu, relu_derivative

def predict(model, x):
    """
    Do a prediction for a single sample
    """
    activation = x.reshape(-1, 1)  # column vector
    for w, b in zip(model['weights'][:-1], model['biases'][:-1]):
        z = np.dot(w, activation) + b
        activation = relu(z)

    # output layer
    z = np.dot(model['weights'][-1], activation) + model['biases'][-1]
    probs = softmax(z)
    return np.argmax(probs)

if __name__ == "__main__":
    # load test data
    X_test, y_test = load_data("data/mnist_test.csv")

    # load trained model
    with open("model_adam_expanded.pkl", "rb") as f:
        model = pickle.load(f)

    # make predictions
    correct = 0
    for i in range(len(X_test)):
        pred = predict(model, X_test[i])
        if pred == y_test[i]:
            correct += 1

        print(f"Predicted: {pred}, Actual: {y_test[i]}")

    accuracy = correct / len(X_test)
    print(f"Test accuracy: {accuracy*100:.2f}% ({correct}/{len(X_test)})")
