import numpy as np
import pickle
import gzip
import random
import matplotlib.pyplot as plt

np.random.seed(1234)
random.seed(1234)

def load_data():
    with gzip.open('mnist.pkl.gz','rb') as f:
        tr_d,va_d,te_d = pickle.load(f,encoding='latin1')
    return tr_d,va_d,te_d

def load_data_wrapper():
    tr_d,va_d,te_d = load_data()
    training_inputs = [np.reshape(x,(1,784)) for x in tr_d[0]]
    training_labels = [vectorize(x) for x in tr_d[1]]
    training_data = zip(training_inputs,training_labels)
    validation_input = [np.reshape(x,(1,784)) for x in va_d[0]]
    validation_data = zip(validation_input,va_d[1])
    test_inputs = [np.reshape(x,(1,784)) for x in te_d[0]]
    test_data = zip(test_inputs,te_d[1])
    return training_data,validation_data,test_data

def vectorize(z):
    j = np.zeros((1,10))
    j[0,z] = 1.0
    return j

training_data,validation_data,test_data = load_data_wrapper()

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1,y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(a,w)+b)
        return a

    def SGD(self,training_data,eta,lamda,mini_batch_size,epochs,test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        training_data = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
        random.shuffle(training_data)
        for epoch in range(epochs):
            for mini_batch in training_data:
                self.update_mini_batch(mini_batch,eta,n,lamda)
            if test_data:
                test_data = list(test_data)
                n1 = len(test_data)
                test_result = self.test_error(test_data,n1)
                print('Epochs:{} Test Accuracy={}/{}'.format(epoch+1,test_result,n1))
        if self.num_layers == 2:
            self.visualize_weight()

    def update_mini_batch(self,mini_batch,eta,n,lamda):
        x = np.squeeze(np.array([x[0] for x in mini_batch]))
        y = np.squeeze(np.array([x[1] for x in mini_batch]))
        activation = x
        activations = [x]
        for b,w in zip(self.biases,self.weights):
            activation = sigmoid(np.dot(activation,w) + b)
            activations.append(activation)
        # backward pass
        loss = (activations[-1]-y)
        delta = loss*(1-activations[-1])*activations[-1]
        grad_b = np.mean(delta,axis=0,keepdims=True)
        self.biases[-1] -= eta*grad_b
        grad_w = np.dot(activations[-2].T,delta)*(1.0/float(len(mini_batch)))
        self.weights[-1] = (1-eta*(lamda)/n)*self.weights[-1] - eta*grad_w
        for l in range(2,self.num_layers):
            delta = np.dot(delta,self.weights[-l+1].T)*(1-activations[-l])*activations[-l]
            grad_b = np.mean(delta,axis=0,keepdims=True)
            self.biases[-l] -= eta*grad_b
            grad_w = np.dot(activations[-l-1].T,delta)*(1.0/float(len(mini_batch)))
            self.weights[-l] = (1-eta*(lamda)/n)*self.weights[-l] - eta*grad_w

    def test_error(self,test_data,n1):
        c = 0
        x_input = np.squeeze(np.array([x[0] for x in test_data]))
        y_label = [x[1] for x in test_data]
        predicted_label = np.argmax(self.feedforward(x_input),axis=1)
        c += np.sum(predicted_label == y_label)
        return c

    def visualize_weight(self):
        for i in range(10):
            #print(self.weights[0].shape)
            plt.imshow(np.reshape(self.weights[0][:,i],(28,28)),cmap='gray')
            plt.show()

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

net = Network([784,100,10])
net.SGD(training_data,2.0,5,64,50,test_data=test_data)
