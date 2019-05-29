import tensorflow as tf

"""
saver = tf.train.Saver()에서
내가 원하는 weight 만 저장하고 싶다면 Saver() 괄호 안에
리스트 형식으로 내가 원하는 weight 만 넣어주면 됨

sampling : 그중에서 가장 좋은 feature을 뽑겠다.

maxpool 할 때 ksize : AxA 에서 최대값 뽑겠다. -> [1,A,A,1]
"""