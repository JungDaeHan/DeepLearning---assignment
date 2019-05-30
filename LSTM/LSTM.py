import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to "
           "collect wood and don't assign them tasks and work, but rather "
           "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i,w in enumerate(char_set)}

dic_size = len(char_dic)
hidden_size = len(char_dic)
num_classes = len(char_dic)
batch_size = 12
sequence_length = 15
learning_rate = 0.1

sample_idx = [char_dic[c] for c in sentence]    
dataX=[]
dataY=[]
i=0

while True:
    
    dataX.append(sample_idx[i:i+sequence_length])
    dataY.append(sample_idx[i+1:i+sequence_length+1])
    
    if i+sequence_length > len(sample_idx):
        dataX.append(sample_idx[i:-1])
        dataY.append(sample_idx[i+1:])
        break
    else : 
        i = i + sequence_length

dataX = dataX[:-2]
dataY = dataY[:-2]
dataY[11].append(0)


tf.reset_default_graph()

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X, num_classes)
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)

X_for_fc = tf.reshape(outputs, [-1,hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        
        # 반복문에서 인덱스 설정 필요없이 알아서 다음 인덱스로 넘어가서 학습해줌
        
        for j in range(len(dataX)):
        
            l, _ = sess.run([loss,train], feed_dict={X: dataX, Y: dataY})
            result = sess.run(prediction, feed_dict={X: dataX})
        
            #result_str = [char_dic[c] for c in np.squeeze(result)]
        
            #print(i, "loss:", l, " Prediction:", ''.join(result_str))
        
    results = sess.run(outputs, feed_dict={X:dataX})

for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j == 0:
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
        