import numpy as np
import tensorflow as tf
import lstm.tf_lstm_cell


def extract_last(x):
    return tf.gather(x, int(x.get_shape()[0]) - 1)


def tf_lstm_copy_of_keras(input_dim, output_dim):
    _lstm_keep_prob = tf.placeholder(tf.float32, (),
                                     name='lstm_keep_prob')

    _x = tf.placeholder(
        dtype=tf.float32, shape=[None, input_dim, 1], name='lstm_1_input')
    _y_true = tf.placeholder(
        dtype=tf.float32, shape=[None, output_dim], name='prediction')

    x_seq = tf.unstack(_x, input_dim, 1)

    with tf.variable_scope('lstm_1'):
        with tf.name_scope('cell_1'):
            cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=50)

        with tf.name_scope('dropout_1'):
            cell_1_drop = tf.contrib.rnn.DropoutWrapper(
                cell_1,
                output_keep_prob=_lstm_keep_prob)

        val_1, state_1 = tf.nn.static_rnn(
            cell=cell_1_drop, inputs=x_seq, dtype=tf.float32)

    # with tf.variable_scope('lstm_2'):
    #     with tf.name_scope('cell_2'):
    #         cell_2 = tf.nn.rnn_cell.LSTMCell(
    #             num_units=100, state_is_tuple=True)
    #
    #     with tf.name_scope('dropout_2'):
    #         cell_2_drop = tf.contrib.rnn.DropoutWrapper(
    #             cell_2,
    #             output_keep_prob=_lstm_keep_prob)
    #
    #     val_2, state_2 = tf.nn.static_rnn(cell_2_drop, val_1, dtype=tf.float32)
    #
    # with tf.variable_scope('lstm_3'):
    #     with tf.name_scope('cell_3'):
    #         cell_3 = tf.nn.rnn_cell.LSTMCell(
    #             num_units=50, state_is_tuple=True)
    #
    #     with tf.name_scope('dropout_3'):
    #         cell_3_drop = tf.contrib.rnn.DropoutWrapper(
    #             cell_3,
    #             output_keep_prob=_lstm_keep_prob)
    #
    #     val_3, state_3 = tf.nn.static_rnn(cell_3_drop, val_2, dtype=tf.float32)

    # with tf.name_scope('multi_rnn'):
    # multi_cell = tf.contrib.rnn.MultiRNNCell([cell_1_drop, cell_2_drop])

    with tf.name_scope('get_last'):
        # Isolate the final value
        last = val_1[-1]

    with tf.name_scope('dense_1'):
        W1 = tf.Variable(tf.truncated_normal([50, output_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[output_dim]))

        y_pred = tf.matmul(last, W1) + b1

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y_pred - _y_true))

    with tf.name_scope('optimiser'):
        optimiser = tf.train.AdamOptimizer(learning_rate=1e-4)

    with tf.name_scope('train_step'):
        train_step = optimiser.minimize(loss)

    return _x, _y_true, y_pred, _lstm_keep_prob, loss, train_step


def tf_lstm_homemade(input_dim, output_dim):
    _x = tf.placeholder(
        dtype=tf.float32, shape=[None, input_dim, 1], name='lstm_1_input')
    _y_true = tf.placeholder(
        dtype=tf.float32, shape=[None, output_dim], name='prediction')

    x_seq = tf.unstack(_x, input_dim, 1)

    with tf.variable_scope('lstm_homemade'):
        val_1, state_1 = lstm.tf_lstm_cell.lstm_homemade(
            units=50, input_sequence=x_seq)

    with tf.name_scope('get_last'):
        # Isolate the final value
        last = val_1[-1]

    with tf.name_scope('dense_1'):
        W1 = tf.Variable(tf.truncated_normal([50, output_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[output_dim]))

        y_pred = tf.matmul(last, W1) + b1

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y_pred - _y_true))

    with tf.name_scope('optimiser'):
        optimiser = tf.train.AdamOptimizer(learning_rate=1e-4)

    with tf.name_scope('train_step'):
        train_step = optimiser.minimize(loss)

    return _x, _y_true, y_pred, loss, train_step


def sample_batch(X, y, batch_size):
    n = len(X)
    ind_n = np.random.choice(n, batch_size, replace=False)
    return list(np.array(X)[ind_n]), list(np.array(y)[ind_n])
