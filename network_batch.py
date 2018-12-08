"""
This code is written by Shashank Shekhar at date 19/08/2018.
In this code the optimization is happen on batch of data at every time batch of data is proceed through neural network and error
is calculated on that batch of data and then weights are updated.I also use regularization.
"""
import numpy as np
import random
import gzip
import pickle
import matplotlib.pyplot as plt

np.random.seed(1234)
random.seed(1234)

def load_data():
    f = gzip.open('mnist.pkl.gz','rb')
    training_data,validation_data,test_data = pickle.load(f,encoding='latin1')
    f.close()
    return (training_data,validation_data,test_data)

def load_data_wrapper():
    tr_d,va_d,te_d = load_data()
    #training_inputs = [np.reshape(x,(28,28)) for i,x in enumerate(tr_d[0]) if i==1]
    #plt.imshow(training_inputs[0],cmap='gray')
    #plt.show()
    #training_labels = [x for i,x in enumerate(tr_d[1]) if i==1]
    #training_data = list(zip(training_inputs,training_labels)) # create a tuple of (x,y) e.g.,[(array([[28,28]],dtype=float32),label)]
    #print(training_data)
    training_inputs = [np.reshape(x,(1,784)) for x in tr_d[0]]
    training_labels = [vectorize(x) for x in tr_d[1]]
    training_data = zip(training_inputs,training_labels)
    validation_inputs = [np.reshape(x,(1,784)) for x in va_d[0]]
    validation_data = zip(validation_inputs,va_d[1])
    test_inputs = [np.reshape(x,(1,784)) for x in te_d[0]]
    test_data = zip(test_inputs,te_d[1])
    return (training_data,validation_data,test_data)

def vectorize(j):
    z = np.zeros((1,10))
    z[0,j] = 1.0
    return z
training_data,validation_data,test_data = load_data_wrapper()

class NetWork(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1,b) for b in self.sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(a,w)+b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,lr,lmbda,test_data=None):
        training_data = list(training_data) # training data in form of zip object so you must have to convert into list or tuple.
        n = len(training_data) #object of type 'zip' has no len()
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)] #'zip' object is not subscriptable
            #[[(1, 2), (2, 3)], [(3, 4), (4, 5)], [(5, 6), (6, 7)], [(7, 8), (8, 9)], [(9, 0), (0, 9)]]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,lr,mini_batch_size,lmbda,n)
            if test_data:
                print('Epochs:{} {}/{}'.format(j,self.evaluate(test_data,j),n_test))
            else:
                print('Epochs:{} Completed!'.format(j))

    def update_mini_batch(self,mini_batch,lr,mini_batch_size,lmbda,n):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # Extract the x_data and y_label
        mini_batch_x_data = np.squeeze(np.array([mini_batch[i][0] for i in range(len(mini_batch))]))
        mini_batch_label = np.squeeze(np.array([mini_batch[i][1] for i in range(len(mini_batch))]))
        print(mini_batch_label)
        # forward pass on that mini_batch
        activation = mini_batch_x_data
        activations = [mini_batch_x_data]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(activation,w)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass on that mini_batch
        delta = (activations[-1]-mini_batch_label)*sigmoid_prime(zs[-1])
        grad_b[-1] = np.sum(delta,axis=0,keepdims=True)*(1.0/float(len(mini_batch)))
        #print(grad_b[-1].shape)
        self.biases[-1] -= lr*grad_b[-1]
        grad_w[-1] = np.dot(activations[-2].T,delta)*(1.0/float(len(mini_batch)))
        #print(grad_w[-1].shape)
        #self.weights[-1] -= lr*grad_w[-1]
        self.weights[-1] = (1-lr*(lmbda/n))*self.weights[-1] - lr*grad_w[-1]
        for l in range(2,self.num_layers):
            delta = np.dot(delta,self.weights[-l+1].T)*sigmoid_prime(zs[-l])
            grad_b[-l] = np.sum(delta,axis=0,keepdims=True)*(1.0/float(len(mini_batch)))
            self.biases[-l] -= lr*grad_b[-l]
            grad_w[-l] = np.dot(activations[-l-1].T,delta)*(1.0/float(len(mini_batch)))
            #print(grad_w[-l].shape)
            #self.weights[-l] -= lr*grad_w[-l]
            self.weights[-l] = (1-lr*(lmbda/n))*self.weights[-l] - lr*grad_w[-l]

    def evaluate(self,test_data,j):
        test_results = [(np.argmax(self.feedforward(x)),y) for x,y in test_data]
        if j == 99:
            for x,y in test_data:
                if np.argmax(self.feedforward(x)) != y:
                    plt.imshow(np.reshape(x,(28,28)),cmap='gray')
                    plt.show()
        return sum(int(x == y) for x,y in test_results)

    def visualize_weight(self):
        #print(self.biases[0])
        #print(self.weights[0])
        for i in range(10):
            #ax = plt.subplot(2,5,i+1)
            #ax.set_title('Weight_of_{}'.format(i))
            #print(np.reshape(np.array(self.weights)[0][:,i],(28,28)))
            plt.imshow(np.reshape(np.array(self.weights)[0][:,i],(28,28)),cmap='gray')
            plt.show()
        #plt.show()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

net = NetWork([784,10])
net.SGD(training_data,20,64,3.0,5.0,test_data=test_data)
net.visualize_weight()
