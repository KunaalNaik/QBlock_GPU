from __future__ import print_function
'''
Basic Multi GPU computation example using TensorFlow library.
Author: Kunaal Naik
'''

'''
This tutorial requires your machine to have 1 GPU
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
'''

import numpy as np
#Import V1 version for compatability 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime

# Processing Units logs
log_device_placement = True

# Num of multiplications to perform
n = 10

'''
Results on 1 GTX-1080 Ti:
 * Single GPU computation time: 0:00:05.405531
'''
# Create random large matrix
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

# Create a graph to store results
c1 = []
c2 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

'''
Single GPU computing
'''
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    # Compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_1 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1-t1_1))
