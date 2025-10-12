# Training

import numpy as np
import pickle

def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    ReLU activation function
    Use RelU for this ANN
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU activation function
    """
    return np.where(x > 0, 1, 0) #  1 when x > 0 else 0, vectorized

def softmax(z):
    """
    Softmax activation function
    """
    max_z = np.max(z, axis=0, keepdims=True)  
    exp_z = np.exp(z - max_z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, sizes):
        """
        Initialize the neural network with given layer sizes
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        # add weight and bias initialization here
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            x = self.sizes[i] 
            y = self.sizes[i + 1] # neurons in next layer
            self.weights.append(np.random.randn(y, x) * np.sqrt(2 / x)) # Kaiming/He initialization
            self.biases.append(np.zeros((y, 1))) # all zeros
            
        

    def forward(self, x):
        """
        Do a forward pass through the network
        """
        self.activations = [x.T]  # store activations for backpropagation (transposed input)
        self.z_values = []        # store z values for backpropagation

        # Hidden layers (all but last)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, self.activations[-1]) + b
            self.z_values.append(z)
            a = relu(z)
            self.activations.append(a)

        # Output layer (softmax)
        w_out = self.weights[-1]
        b_out = self.biases[-1]
        z_out = np.dot(w_out, self.activations[-1]) + b_out
        self.z_values.append(z_out)
        output = softmax(z_out)   # shape (n_classes, batch_size)
        self.activations.append(output)

        return output

    def backward(self, x, y, learning_rate):
        """
        Backpropagation algorithm and weight update
        """
        size = x.shape[0] # batch size
        delta = self.activations[-1] - y.T # error at output 
        nabla_b = [np.zeros(b.shape) for b in self.biases] # gradient for biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # gradient for weights

        # output layer gradients
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, self.activations[-2].T)

        # backpropagate through hidden layers
        for l in range(2, self.num_layers): # start from second last layer and go backwards 
            z = self.z_values[-l]
            derivative = relu_derivative(z) # derivative of activation function
            delta = np.dot(self.weights[-l + 1].T, delta) * derivative # backpropagate the error i.e. update delta
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, self.activations[-l - 1].T)

        # update weights and biases using gradient descent
        for l in range(self.num_layers - 1, 0, -1): # go backwards
            idx = l - 1  # weight/bias index for layer l 
            self.weights[idx] -= (learning_rate / size) * nabla_w[idx]
            self.biases[idx]  -= (learning_rate / size) * nabla_b[idx]


    def train(self, features, targets, learning_rate, num_iterations, batch_size):
        """
        Train the neural network using mini-batch gradient descent
        """
        # TODO: combine forward and backward for mini-batch sgd
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

