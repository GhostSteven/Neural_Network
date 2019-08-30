from __future__ import print_function, division, absolute_import
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt 
import random
import numpy as np
from datasplit import use
#from optparse import OptionParser


testnumber = 4 #要更改的话需要改画图部分的代码否则会出错
path = 'Discuz/'
imgs = os.listdir(path)
model_path = 'model4cnn-1fcn/model.ckpt-500000' #读取你训练好的模型
testdatas = random.sample(imgs,testnumber)
testlabels = list(map(lambda x: x.split('.')[0],testdatas))
#testnum = len(testdatas)
#test_ptr = 0

X = tf.placeholder(tf.float32, [None, 30*100])
Y = tf.placeholder(tf.float32, [None,4*63])
keep_prob = tf.placeholder(tf.float32)

def text2vec(text):
    if len(text) > 4:
        raise ValueError('too long captcha')

    vector = np.zeros(4*63)
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c)-55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')

        return k

    for i, c in enumerate(text):
        idx = i*63 + char2pos(c)
        vector[idx] = 1

    return vector

def vec2text(vec):

    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % 63
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

batch_x = np.zeros([testnumber,30*100])
batch_y = np.zeros([testnumber, 4*63])

for index, test in enumerate(testdatas):
    img = np.mean(cv2.imread(path + test), -1)
    batch_x[index, :] = img.flatten() /255
for index, label in enumerate(testlabels):
    batch_y[index, :] = text2vec(label)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool2d(x, k=2):
    x = tf.nn.max_pool(
        x, ksize=[
            1, k, k, 1], strides=[
            1, k, k, 1], padding='SAME')
    return x

weights = {
        'wc1': tf.Variable(0.01*tf.random_normal([3, 3, 1, 32])),
        'wc2': tf.Variable(0.01*tf.random_normal([3, 3, 32, 64])),
        'wc3': tf.Variable(0.01*tf.random_normal([3, 3, 64, 64])),
        'wc4': tf.Variable(0.01*tf.random_normal([3, 3, 64, 64])),
        'wf1': tf.Variable(0.01*tf.random_normal([2 * 7 * 64, 1024])),
        'wf2': tf.Variable(0.01*tf.random_normal([1024, 1024])),
        'wout': tf.Variable(0.01*tf.random_normal([1024, 4*63]))
        }

biases = {
        'bc1': tf.Variable(0.1*tf.random_normal([32])),
        'bc2': tf.Variable(0.1*tf.random_normal([64])),
        'bc3': tf.Variable(0.1*tf.random_normal([64])),
        'bc4': tf.Variable(0.1*tf.random_normal([64])),
        'bf1': tf.Variable(0.1*tf.random_normal([1024])),
        'bf2': tf.Variable(0.1*tf.random_normal([1024])),
        'bout': tf.Variable(0.1*tf.random_normal([4*63]))
    }

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, [-1,100,30,1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 1)
    conv1 = max_pool2d(conv1, 2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 1)
    conv2 = max_pool2d(conv2, 2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 1)
    conv3 = max_pool2d(conv3, 2)
    
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 1)
    conv4 = max_pool2d(conv4, 2)
    
    fc1 = tf.reshape(
        conv4, shape=[-1, weights['wf1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, weights['wf1'])
    fc1 = tf.add(fc1, biases['bf1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])

    return out

output = conv_net(X, weights, biases, keep_prob)

y = tf.reshape(output, [-1,4,63])
y_ = tf.reshape(Y, [-1,4,63])

predict = tf.argmax(y,2)
correct_pred = tf.equal(predict, tf.argmax(y_,2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)

    pred, acc = sess.run([predict,accuracy], feed_dict ={ X:batch_x, Y:batch_y,keep_prob:1})
    print('accuracy={}'.format(acc))
    for i in range(1,testnumber+1):

        plt.subplot(2,2,i)
        img = cv2.imread(path+testdatas[i-1])
        plt.imshow(img)
        plt.title('number%d' %i)
        plt.xticks([])
        plt.yticks([])
        vect = np.zeros([4*63])

        #print(pred[i-1])
        for ind,j in enumerate(pred[i-1]):
            vect[ind*63+j] = 1

        xlabel = 'True label:{};Pred label:{}'.format(testlabels[i-1], vec2text(vect))
        plt.xlabel(xlabel)

    plt.show()
