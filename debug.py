import numpy as np
import tensorflow as tf
import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(
    input_shape=(1,),
    activation="linear", units=3))

input_feed = [
    np.array([1.]),
    np.array([2.])]

model.predict(np.array(input_feed))

x = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.truncated_normal([1, 3]))
b = tf.Variable(tf.constant(0., shape=[3]))
y_pred = tf.matmul(x, W) + b

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

y_pred.eval(feed_dict={x: input_feed})
