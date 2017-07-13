import numpy as np
import tensorflow as tf
import keras

input_feed = [
    np.array([[1.], [2.]])]

# Keras
model = keras.models.Sequential()
model.add(keras.layers.LSTM(
    units=3,
    input_shape=(2, 1,),
    activation="linear",
    return_sequences=True))
model.predict(np.array(input_feed))

# TensorFlow
x = tf.placeholder(tf.float32, shape=(None, 2, 1))
x_seq = tf.unstack(x, 2, 1)
cell = tf.nn.rnn_cell.LSTMCell(num_units=3)
y_hat, state = tf.nn.static_rnn(cell=cell, inputs=x_seq, dtype=tf.float32)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

[yh.eval(feed_dict={x: input_feed}) for yh in y_hat]
[s.eval(feed_dict={x: input_feed}) for s in state]

x.eval(feed_dict={x: input_feed})
[xs.eval(feed_dict={x: input_feed}) for xs in x_seq]
