import tensorflow as tf
import numpy as np

""" 2013011640 정대한 """

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 16], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(X_img, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 28x28x16 -> 14x14x32
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,16,32], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# 14x14x32 -> 7x7x64
W_conv3 = tf.Variable(tf.truncated_normal(shape=[5,5,32,64], stddev=5e-2))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)

h_conv3 = tf.reshape(h_conv3, [-1, 3136])

W_fc = tf.Variable(tf.truncated_normal(shape=[3136,10]))
b_fc = tf.Variable(tf.constant(0.1, shape=[10]))

logits = tf.matmul(h_conv3,W_fc) + b_fc

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(20):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run([optimizer,cost], feed_dict={X: batch_xs, Y:batch_ys, keep_prob : 0.7})
        total_cost += cost_val
        
    print('Epoch :', '%04d' % (epoch +1),
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
is_correct = tf.equal(tf.arg_max(logits,1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob : 1.0}))

    