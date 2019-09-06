from __future__ import print_function, division
import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
from optparse import OptionParser

mnist = input_data.read_data_sets('MNIST/',one_hot=True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

num_input = 28
timesteps = 28

num_hidden =128
num_classes = 10

X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

W = tf.Variable(tf.random_normal([num_hidden, num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))

cell = tf.nn.rnn_cell.LSTMCell(num_hidden)

parser = OptionParser()
parser.add_option('-d','--dynamic', dest = 'Dynamic_mode', default=False)
opts,args = parser.parse_args()
Dynamic_mode = opts.Dynamic_mode


def dynamic_rnn(cell,x,w,b):
	outputs, states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
	outputs = tf.transpose(tf.concat(outputs,2),[1,0,2])
	return tf.matmul(outputs[-1],w)+b


def static_rnn(cell,x,w,b):
	x = tf.unstack(x,timesteps,1)
	outputs, states = tf.nn.static_rnn(cell,x, dtype = tf.float32)
	return tf.matmul(outputs[-1],w)+b

if Dynamic_mode == True:
	logits = dynamic_rnn(cellX,W,b)
else:
	logits = static_rnn(cell,X,W,b)

print('##############################')
print('Dynamic mode:',Dynamic_mode)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

optimizer  = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)

	for step in range(1,training_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape((batch_size,timesteps, num_input))
		sess.run(train_op,feed_dict = {X:batch_x,Y:batch_y})
		if step % display_step == 0 or step == 1:
			loss, acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y})
			print('step %d, loss = %.4f, Accuracy=%.3f' % (step, loss, acc))

	print('optimizer finished!')

	test_data = mnist.test.images[:batch_size].reshape((-1,timesteps, num_input))
	test_label = mnist.test.labels[:batch_size]
	testAcc = sess.run(accuracy, feed_dict={X:test_data,Y:test_label})
	print('test accuracy = %.3f' % testAcc)

