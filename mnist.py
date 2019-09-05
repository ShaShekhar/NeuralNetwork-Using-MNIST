"""
This code is taken from Micheal Neilson book.I highly recommended to those people who wants to know the mathematics behind the Neural Networks.
In this code the optimization is very slow, because every time single data is proceed through neural network and error is calculated.
This error is accumulated over the batch of data and then weights are updated.
"""
import numpy as np
import random
import gzip
import pickle
import matplotlib.pyplot as plt

# Load the data from pickle 
def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return training_data, validation_data, test_data

# Make the data ready for neural network
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (1 ,784)) for x in tr_d[0]] # reshape the each image for input to the neural network
    training_labels = [vectorize(x) for x in tr_d[1]]         # vectorize the label for computing error, true node 1 and other node 0
    training_data = zip(training_inputs, training_labels)

    validation_inputs = [np.reshape(x, (1, 784)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (1, 784)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return training_data, validation_data, test_data

# one-hot encoding of labels
def vectorize(j):
    z = np.zeros((1, 10))
    z[0, j] = 1.0
    return z

training_data, validation_data, test_data = load_data_wrapper()

class NetWork(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(1, y) for y in self.layers[1:]]
        self.weights = [np.random.randn(x, y) for x,y in zip(self.layers[:-1], self.layers[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(a, w) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data) # randomly shuffle the training data
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('Epochs {} : {}/{}'.format(j, self.evaluate(test_data), n_test))
            else:
                print('Epochs {} completed!'.format(j))

        if (self.num_layers == 2):
            self.visualize_weight()

    def update_mini_batch(self, mini_batch, eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [gb+dgb for gb,dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+dgw for gw,dgw in zip(grad_w, delta_grad_w)]

        self.biases = [b-(eta/len(mini_batch))*gb for b, gb in zip(self.biases, grad_b)]
        self.weights = [w-(eta/len(mini_batch))*gw for w, gw in zip(self.weights, grad_w)]

    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass
        activation = x
        activations = [x] # list to cache the activation for forward pass
        zs = [] # list to cache the weighted input

        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(activations[-2].T,delta)

        for l in range(2,self.num_layers):
            delta = np.dot(delta,self.weights[-l+1].T)*sigmoid_prime(zs[-l])
            grad_b[-l] = delta
            grad_w[-l] = np.dot(activations[-l-1].T,delta)

        return (grad_b, grad_w)

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x,y in test_results)

    def cost_derivative(self, output_activation, y):
        return (output_activation - y)

   # visualize the weight of linear classifier.
    def visualize_weight(self):
        for i in range(10):
            ax = plt.subplot(2,5,i+1)
            ax.set_title('Weight_of_{}'.format(i))
            plt.imshow(np.reshape(np.array(self.weights)[0][:,i],(28,28)),cmap='gray')
            plt.xticks([]),plt.yticks([])
        plt.show()

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

net = NetWork([784,10])
#net = NetWork([784,30,10])
net.SGD(training_data, 30, 64, 2.0, test_data=test_data)
