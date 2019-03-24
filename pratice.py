import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]


hid3dim = tf.Variable(tf.random_normal([128, 1, 150]))
Aout = tf.Variable(tf.random_normal([128, 100, 150]))

# (128, 1, 100)
Aout2dim = tf.matmul(hid3dim, Aout, adjoint_b=True)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(hid3dim))
print("=== b ===")
print(sess.run(Aout2dim))
print("=== expr ===")

Aout2dim = sess.run(Aout2dim, feed_dict={X: x_data})
print(Aout2dim)
print(Aout2dim.shape)

sess.close()
