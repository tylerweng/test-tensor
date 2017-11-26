import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

"""
Constants are initialized when you call tf.constant, and their value can never change. By contrast, variables are not
initialized when you call tf.Variable. To initialize all the variables in a TensorFlow program, you must explicitly
call a special operation as follows:

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables. 
Until we call sess.run, the variables are uninitialized.
"""

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("sess.run(linear_model, {x: [1, 2, 3, 4]})", sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print("sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

print("sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

