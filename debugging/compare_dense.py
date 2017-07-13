import numpy as np
import tensorflow as tf
import keras

input_feed = [
    np.array([1.]),
    np.array([2.])]

# Keras
model = keras.models.Sequential()
model.add(keras.layers.LSTM(
    input_shape=(1,),
    activation="linear", units=3))
model.predict(np.array(input_feed))

# TensorFlow
x = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.truncated_normal([1, 3]))
b = tf.Variable(tf.constant(0., shape=[3]))
y_hat = tf.matmul(x, W) + b

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

y_hat.eval(feed_dict={x: input_feed})
