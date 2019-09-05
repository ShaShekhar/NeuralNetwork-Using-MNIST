import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gzip

tf.set_random_seed(1234)
np.random.seed(1234)

def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data,validation_data,test_data = pickle.load(f,encoding='latin1')

    return training_data, validation_data, test_data

training_data, validation_data, test_data = load_data()

training_inputs, training_labels = training_data[0], training_data[1]
n = len(training_inputs)
training_inputs = [training_inputs[k:k+100] for k in range(0, n, 100)]
training_labels = [training_labels[k:k+100] for k in range(0, n, 100)]
no_of_mini_batch = len(training_inputs)

test_inputs,test_labels = test_data[0],test_data[1]
n1 = len(test_inputs)
test_inputs = [test_inputs[k:k+100] for k in range(0,n1,100)]
test_labels = [test_labels[k:k+100] for k in range(0,n1,100)]
no_of_mini_test = len(test_inputs) #100

class ConvNet(object):
    def __init__(self, lr, sess): # When you pass images and labels since they are placeholder so every time new images are fed new instance are created
        self.lr = lr
        self.sess = sess
        self.image, self.output, self.label, self.keep_prob = self.conv_layers()

        with tf.name_scope('loss'):
            self.xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.output, name='loss'))

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.xent)

        with tf.name_scope('accuracy'):
            self.predicted_output = tf.nn.softmax(self.output)
            self.correct_prediction = tf.equal(tf.argmax(self.predicted_output, axis=1, output_type=tf.int32), self.label)
            self.accuracy = tf.cast(self.correct_prediction, tf.float32)

    def conv_layers(self):
        image = tf.placeholder(tf.float32, shape=[None,784], name='images')
        x_image = tf.reshape(image,[-1,28,28,1])
        tf.summary.image('input', x_image, 3)
        label = tf.placeholder(tf.int32, shape=[None], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('conv1'):
            weight = tf.Variable(tf.truncated_normal(shape=[5,5,1,32], mean=0, stddev=0.01, name='weight'))
            bias = tf.Variable(tf.zeros(shape=[32], name='bias'))
            # Don't make output as instance specific since these value are keep changing
            output = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, weight, strides=[1,1,1,1], padding='SAME'), bias), name='activation')

        with tf.name_scope('pool1'):
            output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool')

        with tf.name_scope('conv2'):
            weight = tf.Variable(tf.truncated_normal(shape=[5,5,32,64], mean=0, stddev=0.01, name='weight'))
            bias = tf.Variable(tf.zeros(shape=[64], name='bias'))
            output = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME'), bias), name='activation')

        with tf.name_scope('pool2'):
            output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool')

        shape = int(np.prod(output.get_shape()[1:]))
        flatten = tf.reshape(output, [-1,shape])  # Create the outside the name scope so it show in form of node name Reshape.

        with tf.name_scope('fc1'):
            weight = tf.Variable(tf.truncated_normal(shape=[shape,1024], mean=0, stddev=0.01, name='weight'))
            bias = tf.Variable(tf.truncated_normal(shape=[1024], mean=0, stddev=0.01, name='bias'))
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten, weight), bias), name='activation')

        output = tf.nn.dropout(output,keep_prob, name='dropout')

        with tf.name_scope('fc2'):
            weight = tf.Variable(tf.truncated_normal(shape=[1024,10], mean=0, stddev=0.01, name='weight'))
            bias = tf.Variable(tf.zeros(shape=[10], name='bias'))
            output = tf.nn.bias_add(tf.matmul(output, weight), bias, name='activation')

        return image,output, label, keep_prob

    def train(self, image, label, keep_prob):
        return self.sess.run(self.optimizer, feed_dict={self.image:image, self.label:label, self.keep_prob:keep_prob}),\
                self.sess.run(self.xent, feed_dict={self.image:image, self.label:label, self.keep_prob:keep_prob})

    def test_accuracy(self, test_image, test_label, keep_prob):
        return self.sess.run(self.accuracy, feed_dict={self.image:test_image, self.label:test_label, self.keep_prob:keep_prob})


with tf.Session() as sess:
    conv = ConvNet(0.001, sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('/tmp/mnist_demo1')
    writer.add_graph(sess.graph)

    for i in range(5):
        print('Epochs:{} Started'.format(i+1))
        for j in range(no_of_mini_batch):
            # Run the training step
            _,loss = conv.train(training_inputs[j],training_labels[j],1.0)

            if j % 50 == 0:
                print('Epochs:{},At Step:{},loss:{:.4f}'.format(i+1,j,loss))

        test_accuracy = []
        for k in range(no_of_mini_test): #100
            part_accuracy = conv.test_accuracy(test_inputs[k], test_labels[k], 1.0)
            test_accuracy.extend(part_accuracy)

        total_test_accuracy = np.sum(test_accuracy)
        print('Test Accuracy {}/{}'.format(total_test_accuracy, n1))

"""
In __init__ method try to keep those value that are constant for all the instance.This idea i get when i put images
and labels in __init__ method.So every time diiferent images are fed a different instance are created i.e. different
Tensorflow graph are created.
"""
