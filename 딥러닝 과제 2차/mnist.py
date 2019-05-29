import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.layers import flatten
#import matplotlib.pylab as plt
#from skimage import util
#import PIL.Image as pilimg

# Set random seed for permutation
np.random.seed(0)
# Set random seed for tf.Variable initialization
tf.set_random_seed(1234)

# Directory(named 'model') for storing trained model.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

print(MODEL_DIR)

""" 2013011640 정대한 """

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

W_fc1 = tf.Variable(tf.truncated_normal(shape=[3136,256]))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]))
h_fc1 = tf.matmul(h_conv3,W_fc1) + b_fc1

W_fc2 = tf.Variable(tf.truncated_normal(shape=[256,10]))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

logits = tf.matmul(h_fc1,W_fc2) + b_fc2
pred = tf.nn.softmax(logits)

is_correct = tf.equal(tf.argmax(logits,1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)


"""
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(10):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run([optimizer,cost], feed_dict={X: batch_xs, Y:batch_ys, keep_prob : 0.5})
        total_cost += cost_val
        
    print('Epoch :', '%04d' % (epoch +1),
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
is_correct = tf.equal(tf.arg_max(logits,1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob : 1.0}))
"""

""" SAVE
saver = tf.train.Saver()
model_path = saver.save(sess, MODEL_DIR + './model.ckpt')
print('Model saved to:', model_path)
print()

"""

#ckpt = tf.train.get_checkpoint_state('C:/Users/lg/.spyder-py3/model')

#new_graph = tf.Graph()
#sess = tf.Session(graph=tf.Graph())
#saver = tf.train.import_meta_graph('C:/Users/lg/.spyder-py3/model/model.ckpt.meta')

#saver = tf.train.Saver()

#saver = tf.train.import_meta_graph('C:/Users/lg/.spyder-py3/model/model.ckpt.meta')
#sess.run(tf.global_variables_initializer())

#saver = tf.train.import_meta_graph('C:/Users/lg/.spyder-py3/model')
#saver.restore(sess,tf.train.latest_checkpoint('./'))

#SAVER_DIR = "C:/Users/lg/.spyder-py3/"
saver = tf.train.Saver()
checkpoint_path = os.path.join(MODEL_DIR, "model")
ckpt = tf.train.get_checkpoint_state(MODEL_DIR)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)    
    print("테스트 데이터 정확도 (Restored) : %f" % accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    sess.close()

# Evaluate accuracy of validation and test datasets
val_acc_t = accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob : 1.0})
print()
print('test accuracy:       {:.4f}'.format(val_acc_t), end='\n')

# A list for f(x): [[], [], [], [], [], [], [], [], [], []]
a = [None]*10
for i in range(10):
    a[i] = list()

# Get f(x) and append to 'a' distinguished by label
label = sess.run(tf.argmax(pred, 1), feed_dict={X: mnist.test.images})
f_x = sess.run(h_fc1, feed_dict={X: mnist.test.images})

for i in range(len(label)):
    a[label[i]].append(np.array(f_x[i]))

data = np.array(a)

a = [None]*10
for i in range(10):
    a[i] = list()
for i in range(10):
    if len(data[i]) == 0:
        a[i] = [0]
    elif len(data[i]) == 1:
        a[i] = data[i][0]
    else:
        for j in range(len(data[i])-2):
            data[i][0] += data[i][j+1]  # 각 클래스별 첫번째 열에 데이터의 합 입력
        a[i] = data[i][0] / len(data[i])

mean = np.array(a)
# print(mean)

tmp = np.array([0.0])

for i in range(10):
    for j in data[i]:
        tmp += np.dot((j-mean[i]),(j-mean[i]).T)
        
tmp = tmp/50000 # (1,) , 400468.54206664

print(np.shape(tmp), tmp)



print(np.shape(f_x), np.cov(f_x))
        

# distance of training datasets
a = [None]*10
for i in range(10):
    a[i] = list()

for i in range(len(label)):
    a[label[i]].append(sum((f_x[i] - mean[label[i]]) ** 2))

distance = np.array(a)

# print(distance[i][int(len(data)*(95.0/100.0))-1])  # 95퍼센트 위치에 해당하는 인덱스의 값(근사)
# print(np.percentile(data, 95))  # 95% 백분위수 출력(데이터의 95%가 발견되는 기댓값)
a = [None]*10
for i in range(10):
    a[i] = list()
    a[i].append(np.percentile(distance[i], 95))

ood_distance = np.array(a)
print("<각 클래스별 ODD distance>")
print(ood_distance)
print()

colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
          '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']

tsne = TSNE(random_state=0)
digit = tsne.fit_transform(f_x)

for i in range(len(f_x)):
    plt.text(digit[i,0], digit[i,1], str(label[i]),
             color = colors[label[i]],
             fontdict={'weight' : 'bold', 'size':9})
    
plt.xlim(digit[:,0].min(), digit[:,0].max())
plt.ylim(digit[:,1].min(), digit[:,1].max())
plt.xlabel('a')
plt.ylabel('b')
plt.show()
    