from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)

max_steps = 3000
batch_size = 128

def variable_with_weight_loss(shape, stddev, wl=None):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)

    return var

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2)
kernel1 = tf.nn.conv2d(image_holder, weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

reshape = tf.reshape(pool2, shape=[batch_size, -1])
dim = reshape.get_shape()[1].value

weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_pre_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    # image_batch, label_batch = sess.run([x_train, y_train])
    # image_batch = tf.train.batch(x_train, batch_size)
    input_queue = tf.train.slice_input_producer([x_train, y_train], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size)
    image_batch, label_batch = sess.run([image_batch, label_batch])

    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_pre_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch')
        print(format_str % (step, loss_value, examples_pre_sec, sec_per_batch))

num_example = 10000
import math

num_iter = int(math.ceil(num_example / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0

while step < num_iter:
    # image_batch, label_batch = sess.run([x_test, y_test])
    input_queue = tf.train.slice_input_producer([x_test, y_test], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size)
    image_batch, label_batch = sess.run([image_batch, label_batch])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

accuracy = true_count / total_sample_count
print('precision @ 1 = %.3f' % accuracy)
