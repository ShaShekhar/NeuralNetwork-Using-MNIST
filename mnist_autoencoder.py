# Implementation of Regular and Denoising Autoencoder.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import gzip
import pickle

# Load the data into memory from pickle file
def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return training_data, validation_data, test_data

# reshape the data for training autoencoder
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    x_train = np.array([np.reshape(x, (28, 28, 1)) for x in tr_d[0]])
    x_val = np.array([np.reshape(x, (28, 28, 1)) for x in va_d[0]])
    x_test = np.array([np.reshape(x, (28, 28, 1)) for x in te_d[0]])

    return x_train, x_val, x_test

x_train, x_val, x_test = load_data_wrapper()

#noise_factor = 0.5
#x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
#x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

#x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
#x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

input_shape = (28, 28,1)
input_layer = tf.keras.Input(shape=input_shape)                     # [None, 28, 28, 1]

x = tf.keras.layers.Conv2D(10, 5, activation='relu')(input_layer)   # (None, 24, 24, 10)
x = tf.keras.layers.MaxPooling2D(2)(x)                              # [None, 12, 12, 10]
x = tf.keras.layers.Conv2D(20, 2, activation='relu')(x)             # [None, 11, 11, 20]
x = tf.keras.layers.MaxPooling2D(2)(x)                              # [None, 5, 5, 20]
encoder = x
x = tf.keras.layers.UpSampling2D(2)(x)                              # [None, 10, 10, 20]
x = tf.keras.layers.Conv2DTranspose(20, 2, activation='relu')(x)    # [None, 11, 11, 20]
x = tf.keras.layers.UpSampling2D(2)(x)                              # [None, 22, 22, 20]
x = tf.keras.layers.Conv2DTranspose(10, 5, activation='relu')(x)    # [None, 26, 26, 10]
x = tf.keras.layers.Conv2DTranspose(1, 3, activation='sigmoid')(x)  # [None, 28, 28, 1]

model = tf.keras.Model(inputs=input_layer, outputs=x)
model.summary()

if os.path.exists('autoencoder.h5'):
    model = tf.keras.models.load_model('autoencoder.h5')
else:
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x_train, x_train, batch_size=32, epochs=2, validation_data=(x_val, x_val))
    model.save('autoencoder.h5')

random_index = np.random.randint(0, 10000, 10)
for i in range(len(random_index)):
    test_sample = x_test[random_index[i]]
    #print(np.array([test_sample]).shape) # (1, 28, 28, 1)
    test_prediction = model.predict(np.array([test_sample])) # model expect the 4d input

    test_sample = np.squeeze(np.array((test_sample * 255), dtype=np.uint8))
    #print(test_sample.shape)
    test_prediction = np.squeeze(np.array((test_prediction * 255), dtype=np.uint8))
    #print(test_prediction.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(test_sample, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(test_prediction, cmap='gray')
    plt.show()
