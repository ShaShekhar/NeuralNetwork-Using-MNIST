import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gzip
import random
import os

#sess = tf.Session()
def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    return training_data, validation_data, test_data

training_data, validation_data, test_data = load_data()

training_inputs, training_labels = training_data[0], training_data[1]
n = len(training_inputs)
training_inputs = [training_inputs[k:k+100] for k in range(0, n, 100)]
training_labels = [training_labels[k:k+100] for k in range(0, n, 100)]
no_of_mini_batch = len(training_inputs)

test_inputs,test_labels = test_data[0],test_data[1]
n1 = len(test_inputs)
test_inputs = [test_inputs[k:k+100] for k in range(0, n1, 100)]
test_labels = [test_labels[k:k+100] for k in range(0, n1, 100)]
no_of_mini_test = len(test_inputs) #100


def conv_layer(inputs, channel_in, channel_out, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=[5,5,channel_in,channel_out], stddev=0.1, dtype=tf.float32, name='W'))
        b = tf.Variable(tf.constant(0.1, shape=[channel_out], name='B'))

        conv = tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding='SAME')
        act = tf.nn.relu(tf.nn.bias_add(conv, b))

        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', act)
        return tf.nn.max_pool(act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def fc_layer(inputs, channel_in, channel_out, name='fc'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=[channel_in,channel_out], stddev=0.01, dtype=tf.float32, name='W'))
        b = tf.Variable(tf.constant(0.1, shape=[channel_out], name='B'))
        if name == 'fc2':
            return tf.nn.bias_add(tf.matmul(inputs,w),b)
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, w), b))
# setup placeholder and reshape the data
x = tf.placeholder(tf.float32,shape=[None,784],name='x')
x_image = tf.reshape(x,[-1,28,28,1])
y = tf.placeholder(tf.int32,shape=[None],name='labels')
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x_image, 1, 32, 'conv1')
conv2 = conv_layer(conv1, 32, 64, 'conv2')

flattened = tf.reshape(conv2, [-1,7*7*64])
fc1 = fc_layer(flattened, 7*7*64, 1024, 'fc1')
# Apply dropout
fc1 = tf.nn.dropout(fc1, keep_prob, name='dropout')
logits = fc_layer(fc1, 1024, 10, 'fc2')

with tf.name_scope('xent'):
    xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

with tf.name_scope('accuracy'):
    predicted_output = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(predicted_output, axis=1, output_type=tf.int32), y)
    accuracy = tf.cast(correct_prediction, tf.float32)

tf.summary.scalar('cross_entropy', xent)
#tf.summary.scalar('accuracy',accuracy)
tf.summary.image('input', x_image, 3)
# How Summaries Work --> Summary op returns protocol buffers and tf.summary.FileWriter writes them to disk
merged_summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    # initialize all the variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./tmp/mnist_demo')
    writer.add_graph(sess.graph)

    if os.path.exists('./mnist_checkpoint'):
        new_saver = tf.train.import_meta_graph('./mnist_checkpoint/my-model-1000.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./mnist_checkpoint'))
    else:
        os.makedirs('./mnist_checkpoint')

    for i in range(20):
        print('Epochs:{} Started'.format(i+1))
        for j in range(no_of_mini_batch):
            # Occasionally report accuracy
            train_dict = {x:training_inputs[j], y:training_labels[j], keep_prob:1.0}

            if j % 5 == 0:
                s = sess.run(merged_summary, feed_dict=train_dict)
                writer.add_summary(s, j)

            # Run the training step
            sess.run(train_step, feed_dict=train_dict)
            loss = sess.run(xent, feed_dict=train_dict)

            if (i+1)*(j+1) % 100 == 0:
                saver.save(sess,'./mnist_checkpoint/my-model', global_step=1000)

            if j % 50 == 0:
                print('Epochs:{}, At Step:{}, loss:{:.4f}'.format(i+1, j, loss))

        test_accuracy = []
        for k in range(no_of_mini_test): #100
            #print(sess.run(logits,feed_dict={x:test_inputs[k]}))
            test_dict = {x:test_inputs[k], y:test_labels[k], keep_prob:1.0}
            part_accuracy = sess.run(accuracy,feed_dict=test_dict)
            test_accuracy.extend(part_accuracy)
        total_test_accuracy = np.sum(test_accuracy)
        print('Test Accuracy {}/{}'.format(total_test_accuracy,n1))

"""
Lession Learned : Don't write loss or accuracy in this way
accuracy = sess.run(accuracy,feed_dict=feed_dict) It gives the Fetch error Since you are fetch same argument again and again
"""
