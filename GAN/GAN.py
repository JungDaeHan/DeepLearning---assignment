# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:07:25 2019

@author: lg
"""

import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(8,8)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28,28))
    return fig

num_img = 0
if not os.path.exists('generated_output/'):
    os.makedirs('generated_output/')

num_epoch = 10000
batch_size = 64
num_input = 28 * 28
num_latent_variable = 100
num_hidden = 128
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, num_input])
z = tf.placeholder(tf.float32, [None, num_latent_variable])

with tf.variable_scope('generator'):
    
    G_W1 = tf.Variable(tf.random_normal(shape=[num_latent_variable, num_hidden], stddev=5e-2))
    G_b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    
    G_W2 = tf.Variable(tf.random_normal(shape=[num_hidden, num_input], stddev=5e-2))
    G_b2 = tf.Variable(tf.constant(0.1, shape=[num_input]))
    
with tf.variable_scope('discriminator'):
    
    D_W1 = tf.Variable(tf.random_normal(shape=[num_input, num_hidden], stddev=5e-2))
    D_b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    
    D_W2 = tf.Variable(tf.random_normal(shape=[num_hidden, 1], stddev=5e-2))
    D_b2 = tf.Variable(tf.constant(0.1, shape=[1]))
    
def build_generator(X):
    hidden_layer = tf.nn.relu((tf.matmul(X, G_W1) + G_b1))
    output_layer = tf.matmul(hidden_layer, G_W2) + G_b2
    generated_mnist_image = tf.nn.sigmoid(output_layer)
    
    return generated_mnist_image

def build_discriminator(X):
    hidden_layer = tf.nn.relu((tf.matmul(X, D_W1) + D_b1))
    logits = tf.matmul(hidden_layer, D_W2) + D_b2
    predicted_value = tf.nn.sigmoid(logits)
    
    return predicted_value, logits

x = mnist.test.images
y = mnist.test.labels
    
    
G = build_generator(z)

D_real, D_real_logits = build_discriminator(X)
D_fake, D_fake_logits = build_discriminator(G)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels = tf.zeros_like(D_fake_logits)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

tvar = tf.trainable_variables()
dvar = [var for var in tvar if 'discriminator' in var.name]
gvar = [var for var in tvar if 'generator' in var.name]

d_train_step = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=dvar)
g_train_step = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=gvar)

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(num_epoch):
        
        batch_X, _ = mnist.train.next_batch(batch_size)
        
        batch_noise = np.random.uniform(-1., 1., [batch_size, 100])
        
        if i % 500 == 0:
            samples = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [64,100])})
            fig = plot(samples)
            plt.savefig('generated_output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 1
            plt.close(fig)
            
        _, d_loss_print = sess.run([d_train_step, d_loss], feed_dict={X: batch_X, z: batch_noise})
        
        _, g_loss_print = sess.run([g_train_step, g_loss], feed_dict={z: batch_noise})
        
        
        if i % 100 == 0:
            print('epoch : %d, Generator 손실함수(g_loss): %f, Discriminator 손실함수(d_loss): %f' % (i, g_loss_print, d_loss_print))
        
        
    
    