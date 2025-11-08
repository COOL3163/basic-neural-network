# Training

import numpy as np
import pickle
from time import perf_counter

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

def one_hot_encode(targets, num_classes=10):
    """
    One-hot encode the labels (thats how the output layer is structured)
    """
    return np.eye(num_classes)[targets] # targets is y

def load_data(path):
    """Load MNIST data from CSV"""
    X, y = [], []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split(",")
            y.append(int(row[0]))
            X.append([float(i) for i in row[1:]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

class NeuralNetwork:
    def __init__(self, sizes, l2_lambda=0.0001):
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
            self.weights.append(np.random.randn(y, x).astype(np.float32) * np.sqrt(2 / x)) # Kaiming/He initialization
            self.biases.append(np.zeros((y, 1), dtype=np.float32)) # all zeros

        # l2 regularization 
        self.l2_lambda = l2_lambda

        # adam optimizer stuff
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.opt_t = 0


    def forward(self, x, scaling_factor=2.0):
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

        z_scaled = z_out * scaling_factor
        self.z_values.append(z_scaled)
        output = softmax(z_scaled)   # shape (n_classes, batch_size)
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

        # # update weights and biases using gradient descent + l2 regularization
        # for idx in range(len(self.weights)):
        #     self.weights[idx] -= learning_rate * (nabla_w[idx] / size + self.l2_lambda * self.weights[idx])
        #     self.biases[idx] -= learning_rate * (nabla_b[idx] / size)

        # Adam optimizer
        beta1 = 0.9
        beta2 = 0.999
        aespsilon = 1e-8
        self.opt_t += 1

        for idx in range(len(self.weights)): # idx = index for weights/biases
            average_w = nabla_w[idx] / size
            average_b = nabla_b[idx] / size

            average_w += self.l2_lambda * self.weights[idx]  # l2 regularization

            # update first moment estimate
            self.m_w[idx] = beta1 * self.m_w[idx] + (1.0 - beta1) * average_w
            self.m_b[idx] = beta1 * self.m_b[idx] + (1.0 - beta1) * average_b

            # update biased second raw moment estimates
            self.v_w[idx] = beta2 * self.v_w[idx] + (1.0 - beta2) * (average_w * average_w)
            self.v_b[idx] = beta2 * self.v_b[idx] + (1.0 - beta2) * (average_b * average_b)

            # compute bias-corrected first and second moments
            m_hat_w = self.m_w[idx] / (1.0 - beta1 ** self.opt_t)
            m_hat_b = self.m_b[idx] / (1.0 - beta1 ** self.opt_t)
            v_hat_w = self.v_w[idx] / (1.0 - beta2 ** self.opt_t)
            v_hat_b = self.v_b[idx] / (1.0 - beta2 ** self.opt_t)

            # update
            self.weights[idx] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + aespsilon)
            self.biases[idx]  -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + aespsilon)
            

    def train(self, features, targets, val_features, val_targets, learning_rate, num_iterations, batch_size, patience=10, decay_rate=0.95, epsilon=1e-8):
        """
        Train the neural network using mini-batch gradient descent
        """
        num_samples = features.shape[0]
        targets_oh = one_hot_encode(targets, num_classes=self.sizes[-1]).astype(np.float32) # one-hot encode targets
        val_targets_oh = one_hot_encode(val_targets, num_classes=self.sizes[-1]).astype(np.float32)

        # early stopping variables
        best_val_loss = float('inf')
        best_weights = None
        best_biases = None
        epochs_without_improvement = 0

        for epoch in range(num_iterations):
            start_time = perf_counter() # start time for epoch
            loss = 0.0 # float

            # learning rate decay
            current_learning_rate = learning_rate * (decay_rate** epoch)

            # shuffle data
            permutation = np.random.permutation(num_samples) # ensure target and features are shuffled the same way
            features_shuffled = features[permutation]
            targets_shuffled = targets_oh[permutation]

            # process mini-batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size # range for mini-batch
                x_batch = features_shuffled[start_idx:end_idx] # features
                y_batch = targets_shuffled[start_idx:end_idx] # targets

                # forward and backward propagation
                self.forward(x_batch)
                self.backward(x_batch, y_batch, current_learning_rate)

                # loss
                loss -= np.sum(y_batch.T * np.log(self.activations[-1] + epsilon))

            loss /= num_samples # average loss
            l2_penalty = sum(np.sum(w ** 2) for w in self.weights)
            loss += self.l2_lambda * l2_penalty
            
            # calculate validation loss
            val_preds = self.forward(val_features)
            val_loss = -np.sum(val_targets_oh.T * np.log(val_preds + epsilon)) / val_features.shape[0]
            val_acc = np.mean(np.argmax(val_preds, axis=0) == val_targets)

            # check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # stop if no improvement for {patience} epochs
            if epochs_without_improvement >= patience:  
                print(f"early stop at epoch {epoch}, best val loss: {best_val_loss:.4f}")
                break

            # debug info
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}, Time = {perf_counter() - start_time:.2f}s")

        # restore best weights and biases
        if best_weights is not None and best_biases is not None:
            self.weights = best_weights
            self.biases = best_biases

    def save(self, path):
        """
        Save the neural network to a file using pickle
        """
        model = {
            "sizes": self.sizes,
            "weights": self.weights,
            "biases": self.biases
        } # dictionary for parameters

        with open(path, 'wb') as f:
            pickle.dump(model, f) # pickle because can directly save python objects ie. no need to convert

if __name__ == "__main__":
    print('loading data')
    features, targets = load_data("data/mnist_train.csv") # load training data
    print(f"laded {features.shape[0]} samples with {features.shape[1]} features each.")
    print('loading val data')
    val_features, val_targets = load_data("data/mnist_val.csv") 
    print(f"laded {val_features.shape[0]} val samples with {val_features.shape[1]} features each.")

    sizes = [features.shape[1], 512, 512, 256, 10] # layer size
    learning_rate = 0.0025
    num_iterations = 128
    batch_size = 100

    print('training model')
    nn = NeuralNetwork(sizes) # initialize the neural network
    nn.train(features, targets, val_features, val_targets, learning_rate, num_iterations, batch_size=batch_size) # train the model
    
    nn.save('model_final1.pkl') # save the model

