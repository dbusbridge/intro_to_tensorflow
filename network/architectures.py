import tensorflow as tf
import network.variables as variables


def linear_regression(device='/cpu:0'):
    """Create a neural network that has an architecture that matches linear
    regression.
    :param str device: The device to use for storing variables and computation.
        Either '/cpu:<n>' or '/cpu:<n>'. Defaults to '/cpu:<n>.
    :return: A set of TensorFlow tensors that serve as inputs, controllers and
        outputs of the network.
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor).
    """
    with tf.device(device_name_or_function=device):
        # Placeholders for feed and output
        x = tf.placeholder(tf.float32, shape=[None, 1])
        y_true = tf.placeholder(tf.float32, shape=[None, 1])

        # First layer: Weights + bias
        w = variables.weight(shape=[1])
        b = variables.bias(shape=[1])

        # Readout layer: perform linear transformation + bias with relu
        y_pred = tf.multiply(x, w) + b

    return x, y_true, y_pred, w, b


def feed_forward_nn(device='/cpu:0',
                    input_layer_size=1,
                    output_layer_size=1,
                    n_neurons=50,
                    end_activation=None):
    """Create a neural network that has one hidden layer.
    :param str device: The device to use for storing variables and computation.
        Either '/cpu:<n>' or '/cpu:<n>'. Defaults to '/cpu:<n>.
    :return: A set of TensorFlow tensors that serve as inputs, controllers and
        outputs of the network.
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor).
    """
    with tf.device(device_name_or_function=device):
        # Placeholders for feed and output
        x = tf.placeholder(tf.float32, shape=[None, input_layer_size])
        y_true = tf.placeholder(tf.float32, shape=[None, output_layer_size])

        # First layer: Weights + bias
        w1 = variables.weight(shape=[input_layer_size, n_neurons])
        b1 = variables.bias(shape=[n_neurons])

        # Readout layer: perform linear transformation + bias with relu
        if input_layer_size == 1:
            h1 = tf.nn.relu(tf.multiply(x, w1) + b1)
        else:
            h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        # Second layer: Weights + bias
        w2 = variables.weight(shape=[n_neurons, output_layer_size])
        b2 = variables.bias(shape=[output_layer_size])

        if end_activation is None:
            y_pred = tf.matmul(h1, w2) + b2
        else:
            y_pred = end_activation(tf.matmul(h1, w2) + b2)

    return x, y_true, y_pred, w1, w2, b1, b2
