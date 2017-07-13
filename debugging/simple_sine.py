import responses

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import lstm.tf

from sklearn.model_selection import TimeSeriesSplit

# Global configuration
X_RANGE_MIN, X_RANGE_MAX = -2 * np.pi, 2 * np.pi
N = 256
TEST_SIZE = 0.5
PAST_SEQUENCE_LENGTH = 10
FUTURE_SEQUENCE_LENGTH = 1
STEPS_AHEAD = 1

# Create the data
X = np.arange(start=X_RANGE_MIN,
              stop=X_RANGE_MAX,
              step=(X_RANGE_MAX-X_RANGE_MIN)/N)
# np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
X = np.sort(X)
y = responses.general(X, f=lambda a: np.sin(a), noise_sd=0.0)

n_sequences = len(X) - np.max([PAST_SEQUENCE_LENGTH,
                               FUTURE_SEQUENCE_LENGTH]) - STEPS_AHEAD


# Build the time series
def sub_sequence(x, i, sequence_length):
    return x[i:i + sequence_length]

X_past_sequences = np.array([sub_sequence(
    x=X, i=i,
    sequence_length=PAST_SEQUENCE_LENGTH)
    for i in range(n_sequences)])
y_past_sequences = np.array([sub_sequence(
    x=y, i=i,
    sequence_length=PAST_SEQUENCE_LENGTH)
    for i in range(n_sequences)])

X_future_sequences = np.array([sub_sequence(
    x=X, i=i + PAST_SEQUENCE_LENGTH + (STEPS_AHEAD - 1),
    sequence_length=FUTURE_SEQUENCE_LENGTH)
    for i in range(n_sequences)])

y_future_sequences = np.array([sub_sequence(
    x=y, i=i + PAST_SEQUENCE_LENGTH + (STEPS_AHEAD - 1),
    sequence_length=FUTURE_SEQUENCE_LENGTH)
    for i in range(n_sequences)])

# Build training testing sample
tscv = TimeSeriesSplit(n_splits=2)
for train_index, test_index in tscv.split(X_past_sequences):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_past_sequences_train = X_past_sequences[train_index]
    X_past_sequences_test = X_past_sequences[test_index]

    X_future_sequences_train = X_future_sequences[train_index]
    X_future_sequences_test = X_future_sequences[test_index]

    y_past_sequences_train = y_past_sequences[train_index]
    y_past_sequences_test = y_past_sequences[test_index]

    y_future_sequences_train = y_future_sequences[train_index]
    y_future_sequences_test = y_future_sequences[test_index]

# Let's try one-dimensional
y_train_input = list(
    np.reshape(y_past_sequences_train,
               tuple(list(y_past_sequences_train.shape) + [1])))
y_train_output = list(
    np.reshape(y_future_sequences_train,
               tuple(list(y_future_sequences_train.shape))))

y_test_input = list(
    np.reshape(y_past_sequences_test,
               tuple(list(y_past_sequences_test.shape) + [1])))
y_test_output = list(
    np.reshape(y_future_sequences_test,
               tuple(list(y_future_sequences_test.shape))))

x = tf.placeholder(tf.float32, shape=(None, PAST_SEQUENCE_LENGTH, 1))
y_true = tf.placeholder(
    dtype=tf.float32, shape=[None, FUTURE_SEQUENCE_LENGTH], name='prediction')

x_seq = tf.unstack(x, PAST_SEQUENCE_LENGTH, 1)
cell = tf.nn.rnn_cell.LSTMCell(num_units=16)
output, state = tf.nn.static_rnn(cell=cell, inputs=x_seq, dtype=tf.float32)

final_output = output[-1]

W = tf.Variable(tf.truncated_normal([16, 1]))
b = tf.Variable(tf.constant(0., shape=[1]))
y_hat = tf.matmul(final_output, W) + b

loss = tf.reduce_mean(tf.square(y_hat - y_true))
optimiser = tf.train.AdamOptimizer(learning_rate=1e-4)
train_step = optimiser.minimize(loss)

STEPS = 10000
PRINT_INVERSE_FREQ = 100

# TensorFlow training
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for s in range(STEPS):
    if s % PRINT_INVERSE_FREQ == 0:
        train_loss = loss.eval(feed_dict={x: y_train_input,
                                          y_true: y_train_output})

        val_loss = loss.eval(feed_dict={x: y_test_input,
                                        y_true: y_test_output})

        msg = "step: {e}, loss: {tr_e}, val_loss: {ts_e}".format(
            e=s, tr_e=train_loss, ts_e=val_loss)

        print(msg)

    x_batch, y_batch = y_train_input, y_train_output

    feed_dict = {x: x_batch,
                 y_true: y_batch}

    sess.run(train_step, feed_dict=feed_dict)


y_future_sequences_train_predictions_tf = y_hat.eval(
    feed_dict={x: y_train_input})

y_future_sequences_test_predictions_tf = y_hat.eval(
    feed_dict={x: y_test_input})

plt.scatter(X_future_sequences_train, y_future_sequences_train,
            color='black', label='training')
plt.scatter(X_future_sequences_test, y_future_sequences_test,
            color='red', label='test')

plt.xlabel("X")
plt.ylabel("y")

plt.scatter(X_future_sequences_train,
            y_future_sequences_train_predictions_tf,
            color='cyan', label='training - predicted (TensorFlow)')

plt.scatter(X_future_sequences_test,
            y_future_sequences_test_predictions_tf,
            color='orange', label='test - predicted (TensorFlow)')

plt.legend()

plt.title("Clearly I don't understand something...")
