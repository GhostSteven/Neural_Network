from __future__ import print_function, division, absolute_import
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt 
import random
import numpy as np
from optparse import OptionParser

path = 'Discuz/' #存放数据的路径
imgs = os.listdir(path) #以列表形式读取所有图片名称
random.shuffle(imgs) #打乱
max_steps = 1000000 #最大迭代步数
save_path = 'model4cnn-1fcn' #保存模型的路径，会自动生成
dropout = 1 #没用到

trainnum = 50000 #定义训练集和测试集的大小
testnum = 10000

traindatas = imgs[:trainnum] #取出训练集和测试集及其标签
trainlabels = list(map(lambda x: x.split('.')[0],traindatas))

testdatas = imgs[trainnum:]
testlabels = list(map(lambda x: x.split('.')[0],testdatas))

#定义取数据集的指针
train_ptr = 0
test_ptr = 0

def next_batch(batch=100, train_flag=True):
    global train_ptr
    global test_ptr
    batch_x = np.zeros([batch,30*100])
    batch_y = np.zeros([batch, 4*63])

    if train_flag == True:
        if batch + train_ptr < trainnum:
            trains = traindatas[train_ptr:(train_ptr+batch)]
            labels = trainlabels[train_ptr:(train_ptr+batch)]
            train_ptr += batch
        else:
            new_ptr = (train_ptr + batch) % trainnum 
            trains = traindatas[train_ptr:] + traindatas[:new_ptr]
            labels = trainlabels[train_ptr:] + traindatas[:new_ptr]
            train_ptr = new_ptr

        for index, train in enumerate(trains):
            img = np.mean(cv2.imread(path + train), -1)
            batch_x[index,:] = img.flatten() /255
        for index, label in enumerate(labels):
            batch_y[index,:] = text2vec(label)

    else:
        if batch + test_ptr < testnum:
            tests = testdatas[test_ptr:(test_ptr+batch)]
            labels = testlabels[test_ptr:(test_ptr+batch)]
            test_ptr += batch
        else:
            new_ptr = (test_ptr + batch) % testnum 
            tests = testdatas[test_ptr:] + testdatas[:new_ptr]
            labels = testlabels[test_ptr:] + testlabels[:new_ptr]
            test_ptr = new_ptr

        for index, test in enumerate(tests):
            img = np.mean(cv2.imread(path + test), -1)
            batch_x[index, :] = img.flatten() /255
        for index, label in enumerate(labels):
            batch_y[index,:] = text2vec(label)

    return batch_x, batch_y

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

X = tf.placeholder(tf.float32, [None, 30*100])
Y = tf.placeholder(tf.float32, [None,4*63])
_lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

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

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=output, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=_lr).minimize(loss_op)

y = tf.reshape(output, [-1,4,63])
y_ = tf.reshape(Y, [-1,4,63])

correct_pred = tf.equal(tf.argmax(y, 2), tf.argmax(y_,2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
lr = 0.001
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1,1+max_steps):
        batch_x, batch_y = next_batch(100,True)
        loss_value,_ = sess.run([loss_op, optimizer],
            feed_dict = {X:batch_x, Y:batch_y, keep_prob:dropout,_lr:lr})
        if step % 10 == 0:
            batch_x_test, batch_y_test = next_batch(100, False)
            acc = sess.run(accuracy, 
                feed_dict={X:batch_x_test, Y:batch_y_test,keep_prob:1})
            print('step{}, loss={}, accuracy={}'.format(step,loss_value, acc))

        if step % 500 == 0:
            random.shuffle(traindatas)
            trainlabels = list(map(lambda x: x.split('.')[0],traindatas))

        if step % 3000 == 0:
            lr *= 0.9

        if step % 10000 == 0:
            saver.save(sess, save_path + "/model.ckpt-%d" % step)
            print('model saved!')
