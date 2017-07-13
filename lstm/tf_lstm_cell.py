import tensorflow as tf


# sequence_length = 2
# sequence_dimension = 3
# units = 10
#
# x = tf.placeholder(
#     dtype=tf.float32, shape=(None, sequence_length, sequence_dimension))
# x_seq = tf.unstack(x, sequence_length, 1)
def lstm_homemade(units, input_sequence):
    zeros_dims = tf.stack([tf.shape(input_sequence[0])[0], units])
    state_sequence = [tf.fill(zeros_dims, 0.0)]
    output_sequence = [tf.fill(zeros_dims, 0.0)]

    layer_names = ('forget_gate', 'input_gate', 'output_gate', 'input_layer')
    x_size = int(input_sequence[0].get_shape()[1])

    b, W, U = {}, {}, {}
    for layer_name in layer_names:
        with tf.name_scope(layer_name):
            b[layer_name] = tf.Variable(
                initial_value=tf.constant(0.1, shape=[units]),
                name='b',
                dtype=tf.float32)

            W[layer_name] = tf.Variable(
                initial_value=tf.truncated_normal([x_size, units]),
                name='W')

            U[layer_name] = tf.Variable(
                initial_value=tf.truncated_normal([units, units]),
                name='U')

    for t, x_t in enumerate(input_sequence):
        with tf.name_scope('time_step_{t}'.format(t=t)):
            s_t_m1 = state_sequence[t]
            h_t_m1 = output_sequence[t]

            gate_value = {g: tf.nn.sigmoid(
                b[g] + tf.matmul(x_t, W[g]) + tf.matmul(h_t_m1, U[g]))
                for g in ('forget_gate', 'input_gate', 'output_gate')}

            s_t = (gate_value['forget_gate'] * s_t_m1 +
                   gate_value['input_gate'] * tf.nn.sigmoid(
                       b['input_layer'] +
                       tf.matmul(x_t, W['input_layer']) +
                       tf.matmul(h_t_m1, U['input_layer'])))

            h_t = tf.nn.tanh(s_t) * gate_value['output_gate']

            state_sequence.append(s_t)
            output_sequence.append(h_t)

    return output_sequence[1:], state_sequence[1:]
#
# values, states = lstm(units=units, input_sequence=x_seq)
#
# writer = tf.summary.FileWriter('graphs/dan_lsm')
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
# writer.add_graph(sess.graph)
