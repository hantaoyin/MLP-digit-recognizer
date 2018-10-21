#!/usr/bin/env python3

# Modified from tensorflow tutorial:
# https://www.tensorflow.org/tutorials/keras/basic_classification
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

img_width = 28
img_height = 28

train_file = open('data/train.csv')
test_file = open('data/test.csv')

train_labels = []
train_images = []
test_images = []

next(train_file)
for line in train_file:
    v = [int(a) for a in line.split(',')]
    assert(len(v) == 785)
    arr = np.reshape(np.array(v[1:]) / 255.0, [img_height, img_width])
    label = np.full([10], 1.e-5)
    label[v[0]] = 1. - 1.e-5 * 9
    train_images.append(arr.astype(np.float32))
    train_labels.append(label.astype(np.float32))
    # Enlarge training set by generate translate images (using periodic boundary conditions).
    # for i in range(10):
    #     train_labels.append(v[0])
    #     translated = np.roll(arr,
    #                          shift=(random.randrange(0, img_height), random.randrange(0, img_width)),
    #                          axis=(0, 1))
    #     train_images.append(translated)

train_labels = np.stack(train_labels)
train_images = np.stack(train_images)

next(test_file)
for line in test_file:
    v = [int(a) for a in line.split(',')]
    assert(len(v) == 784)
    arr = np.reshape(np.array(v) / 255.0, [img_height, img_width])
    test_images.append(arr.astype(np.float32))

test_images = np.stack(test_images)

# Note: Somehow using the default alpha (the coefficient for negative input) in LeakyReLU() causes
# the loss to break down after a few epochs (no matter how large I set the regularizer coefficient).
# I haven't figured out why, but my guess is that coefficients in network drift and diverges in such
# scenarios. Instead, it seems reducing the value of alpha makes it more stable (but doesn't
# eliminate the problem).
def build_network_model():
    regularizer = tf.keras.regularizers.l2(1.e-4)
    def keras_residual_block(input_layer):
        v = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(input_layer)
        v = tf.keras.layers.Conv2D(128, 3, data_format='channels_last', kernel_regularizer=regularizer)(v)
        v = tf.keras.layers.LeakyReLU(0.01)(v)
        v = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(v)
        v = tf.keras.layers.Conv2D(128, 3, data_format='channels_last', kernel_regularizer=regularizer)(v)
        v = tf.keras.layers.Add()([input_layer, v])
        return tf.keras.layers.LeakyReLU(0.01)(v)

    inputs = tf.keras.layers.Input(shape=([img_height, img_width]))
    v = tf.keras.layers.Reshape(target_shape=[img_height, img_width, 1])(inputs)
    v = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(v)
    v = tf.keras.layers.Conv2D(128, 3, data_format='channels_last', kernel_regularizer=regularizer)(v)
    v = tf.keras.layers.LeakyReLU(0.01)(v)
    for i in range(3):
        v = keras_residual_block(v)
    v = tf.keras.layers.Flatten(input_shape=[img_height, img_width, 128])(v)
    v = tf.keras.layers.Dense(128, kernel_regularizer=regularizer)(v)
    v = tf.keras.layers.LeakyReLU(0.01)(v)
    v = tf.keras.layers.Dense(10, kernel_regularizer=regularizer)(v)
    v = tf.keras.layers.Softmax()(v)
    return tf.keras.models.Model(inputs=[inputs], outputs=[v])

# ====================================================================================================
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=[img_height, img_width]),
#     tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax),
# ])
# model = tf.keras.Sequential([
#     tf.keras.layers.Reshape(target_shape=[img_height, img_width, 1]),
#     tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, data_format='channels_last'),
#     keras_residual_block,
#     # tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, data_format='channels_last'),
#     tf.keras.layers.Flatten(input_shape=[img_height, img_width, 128]),
#     tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax),
# ])
model = build_network_model()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
predictions = model.predict(test_images)

def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)

    plt.xlabel('{} {:2.0f}% ({})'.format(predicted_label,
                                         100 * np.max(predictions_array),
                                         predicted_label,
                                         color='blue'))

def plot_value_array(i, predictions_array):
    predictions_array = predictions_array[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('blue')

def gen_submission_file(predictions_array):
    f = open('submission.csv', 'w')
    f.write('ImageId,Label\n')
    for i in range(len(predictions_array)):
        f.write('{},{}\n'.format(i + 1, np.argmax(predictions_array[i])))
    f.close()

gen_submission_file(predictions)

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2 * i + 1)
    plot_image(i, predictions, test_images)
    plt.subplot(num_rows, 2*num_cols, 2 * i + 2)
    plot_value_array(i, predictions)

plt.show()
