import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, sizes):
        """
        Initialize the neural network with given layer sizes
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        # add weight and bias initialization here

    def forward(self, x):
        """
        Do a forward pass through the network
        """
        pass

    def backward(self, x, y, learning_rate):
        """
        Backpropagation algorithm and weight update
        """
        pass

    def train(self, features, targets, learning_rate, num_iterations, batch_size):
        """
        Train the neural network using mini-batch gradient descent
        """
        pass

    def save(self, path):
        """
        Save the neural network to a file using pickle
        """
        model = {} # dictionary for parameters

        with open(path, 'wb') as f:
            pickle.dump(model, f) # pickle because can directly save python objects ie. no need to convert

if __name__ == "__main__":
    sizes = [] # layer size
    nn = NeuralNetwork(sizes) # initialize the neural network

    nn.save('model.pkl') # save the model

