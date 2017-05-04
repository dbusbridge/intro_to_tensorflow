import tensorflow as tf

# Directory to log TensorBoard to
log_dir = 'logs/addition/'


# Simple function for addition
def add(a, b):
    return a + b


# Graph of addition function with inputs
# Change shape=() -> shape=(None, ) for batching
def add_tf_graph():
    with tf.name_scope('inputs'):
        a = tf.placeholder(tf.float32, shape=(), name="a")
        b = tf.placeholder(tf.float32, shape=(), name="b")

    with tf.name_scope('outputs'):
        r = tf.add(a, b, name="r")

    return a, b, r

# Construct a specific instance of the addition graph
a_input, b_input, a_plus_b = add_tf_graph()

# Initialise the session and the variables on the session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Print if the results of the computation are the same
result = sess.run(a_plus_b, feed_dict={a_input: 2, b_input: 5})

writer = tf.summary.FileWriter(log_dir)
writer.add_graph(sess.graph)

print(result == add(2., 5.))
