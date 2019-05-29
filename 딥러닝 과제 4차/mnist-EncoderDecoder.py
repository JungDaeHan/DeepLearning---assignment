import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt 

""" 2013011640 정대한 """

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 784])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([784,256], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([256], -1.0, 1.0))
L1 = tf.nn.sigmoid(tf.matmul(X,W1) + b1)
L1 = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.random_uniform([256,64], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([64], -1.0, 1.0))
L2 = tf.nn.sigmoid(tf.matmul(L1,W2) + b2)
L2 = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.random_uniform([64,256], -1.0, 1.0))
b3 = tf.Variable(tf.random_uniform([256], -1.0, 1.0))
L3 = tf.nn.sigmoid(tf.matmul(L2,W3) + b3)
L3 = tf.nn.dropout(L3,keep_prob)

W4 = tf.Variable(tf.random_uniform([256,784], -1.0, 1.0))
b4 = tf.Variable(tf.random_uniform([784], -1.0, 1.0))

logits = tf.matmul(L3,W4) + b4
model = tf.nn.sigmoid(logits)

cost = tf.reduce_mean(tf.squared_difference(model,Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.05).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(20):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, _= mnist.train.next_batch(batch_size)
        
        noise_batch_xs = batch_xs + np.random.uniform(0,0.1,[100,784])
        
        _, cost_val,output= sess.run([optimizer,cost,model], feed_dict={X: noise_batch_xs, Y:batch_xs, keep_prob : 0.7})
        total_cost += cost_val
        
        #out = sess.run(model, feed_dict={X:mnist.test.images[0],keep_prob:1.0})
        
        #plt.figure(figsize=(5,5))
        #plt.imshow(np.reshape(out,[28,28]), cmap='Greys')
        #plt.show()
        
    print('Epoch :', '%04d' % (epoch +1),
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

for i in range(10):    
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(mnist.test.images[i:i+1], [28, 28]), cmap='Greys') # (100,784)
    plt.show()
    
    sam = sess.run(model, feed_dict={ X:mnist.test.images[i:i+1], Y:mnist.test.images[i:i+1] , keep_prob:1.0})
    
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(sam,[28,28]), cmap='Greys')
    plt.show()
        
    
    #print(sess.run(output, feed_dict={X: x_data}))   
    