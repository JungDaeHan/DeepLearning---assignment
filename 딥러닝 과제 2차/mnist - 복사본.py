import tensorflow as tf
import numpy as np

""" 2013011640 정대한 """

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_uniform([784,256], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([256], -1.0, 1.0))
L1 = tf.nn.relu(tf.matmul(X,W1) + b1)

W2 = tf.Variable(tf.random_uniform([256,256], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([256], -1.0, 1.0))
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2)

W3 = tf.Variable(tf.random_uniform([256,10], -1.0, 1.0))
b3 = tf.Variable(tf.random_uniform([10], -1.0, 1.0))
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3)

logits = tf.matmul(L2,W3) + b3
output_softmax = tf.nn.relu(logits)
output_argmax = tf.argmax(logits, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run([optimizer,cost], feed_dict={X: batch_xs, Y:batch_ys})
        total_cost += cost_val
        
    print('Epoch :', '%04d' % (epoch +1),
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
is_correct = tf.equal(tf.arg_max(logits,1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    
    #print(sess.run(output, feed_dict={X: x_data}))   
    