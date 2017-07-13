import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit

import responses
# import lstm.keras
import lstm.tf

# Global configuration
X_RANGE_MIN, X_RANGE_MAX = -8 * np.pi, 8 * np.pi
N = 1024 * 4
TEST_SIZE = 0.33
PAST_SEQUENCE_LENGTH = 20
FUTURE_SEQUENCE_LENGTH = 1
STEPS_AHEAD = 20

tomas_func = lambda a: -(1 + np.exp(-0.5 * a)) * np.arctan(np.sin(a)/(1+np.exp(-0.5 * a) - np.cos(a)))
dan_func = lambda a: a ** 2
funcy_pidgeon = lambda a: a * np.sin(a)

# Create the data
X = np.arange(start=X_RANGE_MIN,
              stop=X_RANGE_MAX,
              step=(X_RANGE_MAX-X_RANGE_MIN)/N)
# np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
X = np.sort(X)
y = responses.general(X, f=lambda a: a * np.sin(a), noise_sd=0.0)

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

# For plotting
# train_plot_subset = X <= np.max(X_future_sequences_train)
# test_plot_subset = X > np.max(X_future_sequences_train)
# X_plot_train, X_plot_test = X[train_plot_subset], X[test_plot_subset]
# y_plot_train, y_plot_test = y[train_plot_subset], y[test_plot_subset]

# Of course we could train a function f(x) = yhat ~ x * cos(x)
# But could we do this with a sequence learner, e.g. an LSTM?

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

num_units = PAST_SEQUENCE_LENGTH
lstm_keep_prob = 1

EPOCH_SIZE = len(y_train_input)
EPOCHS = 1024 * 16
BATCH_SIZE = 256
STEPS = int(EPOCH_SIZE * EPOCHS / BATCH_SIZE)
PRINT_INVERSE_FREQ = 50

# Keras model
# keras_lstm = lstm.keras.keras_lstm(input_dim=PAST_SEQUENCE_LENGTH,
#                                    output_dim=FUTURE_SEQUENCE_LENGTH)

# TensorFlow model
(_x, _y_true, y_pred, _lstm_keep_prob,
 loss, train_step) = lstm.tf.tf_lstm_copy_of_keras(
    input_dim=PAST_SEQUENCE_LENGTH,
    output_dim=FUTURE_SEQUENCE_LENGTH)

(_x_h, _y_true_h, y_pred_h, loss_h, train_step_h) = lstm.tf.tf_lstm_homemade(
    input_dim=PAST_SEQUENCE_LENGTH,
    output_dim=FUTURE_SEQUENCE_LENGTH)


# # Keras training
# keras_lstm.fit(
#     x=np.array(y_train_input),
#     y=np.array(y_train_output),
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_split=0.05)

# TensorFlow training
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("Starting TensorFlow implementation")
for s in range(STEPS):
    if s % PRINT_INVERSE_FREQ == 0:
        train_loss = loss.eval(feed_dict={_x: y_train_input,
                                          _y_true: y_train_output,
                                          _lstm_keep_prob: 1})

        val_loss = loss.eval(feed_dict={_x: y_test_input,
                                        _y_true: y_test_output,
                                        _lstm_keep_prob: 1})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e}".format(
            e=s, tr_e=train_loss, ts_e=val_loss, steps=STEPS)

        print(msg)

    x_batch, y_batch = lstm.tf.sample_batch(X=y_train_input,
                                            y=y_train_output,
                                            batch_size=BATCH_SIZE)

    feed_dict = {_x: x_batch,
                 _y_true: y_batch,
                 _lstm_keep_prob: lstm_keep_prob}

    sess.run(train_step, feed_dict=feed_dict)

print("Starting homemade TensorFlow implementation")
for s in range(STEPS):
    if s % PRINT_INVERSE_FREQ == 0:
        train_loss = loss_h.eval(feed_dict={_x_h: y_train_input,
                                          _y_true_h: y_train_output})

        val_loss = loss_h.eval(feed_dict={_x_h: y_test_input,
                                        _y_true_h: y_test_output})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e}".format(
            e=s, tr_e=train_loss, ts_e=val_loss, steps=STEPS)

        print(msg)

    x_batch, y_batch = lstm.tf.sample_batch(X=y_train_input,
                                            y=y_train_output,
                                            batch_size=BATCH_SIZE)

    feed_dict = {_x_h: x_batch,
                 _y_true_h: y_batch}

    sess.run(train_step_h, feed_dict=feed_dict)

# y_future_sequences_train_predictions_keras = keras_lstm.predict(
#     np.array(y_train_input))
#
# y_future_sequences_test_predictions_keras = keras_lstm.predict(
#     np.array(y_test_input))

y_future_sequences_train_predictions_tf = y_pred.eval(
    feed_dict={_x: y_train_input,
               _lstm_keep_prob: 1})

y_future_sequences_test_predictions_tf = y_pred.eval(
    feed_dict={_x: y_test_input,
               _lstm_keep_prob: 1})

y_future_sequences_train_predictions_tf_h = y_pred_h.eval(
    feed_dict={_x_h: y_train_input})

y_future_sequences_test_predictions_tf_h = y_pred_h.eval(
    feed_dict={_x_h: y_test_input})

plt.scatter(X_future_sequences_train, y_future_sequences_train,
            color='black', label='training')
plt.scatter(X_future_sequences_test, y_future_sequences_test,
            color='red', label='test')
plt.xlabel("X")
plt.ylabel("y")
#
# plt.scatter(X_future_sequences_train,
#             y_future_sequences_train_predictions_keras,
#             color='blue', label='training - predicted (Keras)')
#
# plt.scatter(X_future_sequences_test,
#             y_future_sequences_test_predictions_keras,
#             color='green', label='test - predicted (Keras)')

plt.scatter(X_future_sequences_train,
            y_future_sequences_train_predictions_tf,
            color='cyan', label='training - predicted (TensorFlow)')

plt.scatter(X_future_sequences_test,
            y_future_sequences_test_predictions_tf,
            color='orange', label='test - predicted (TensorFlow)')

plt.scatter(X_future_sequences_train,
            y_future_sequences_train_predictions_tf_h,
            color='blue', label='training - predicted (TensorFlow homemade)')

plt.scatter(X_future_sequences_test,
            y_future_sequences_test_predictions_tf_h,
            color='green', label='test - predicted (TensorFlow homemade)')

plt.legend()

plt.title("Clearly I don't understand something...")
