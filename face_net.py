from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import input_data

IMAGE_SIZE = input_data.IMAGE_SIZE
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
CHANNELS = 3

CONV_SIZE = 5
CONV1_LAYER = 64
CONV2_LAYER = 64

POOLED_IMAGE_SIZE = IMAGE_SIZE // 4
POOLED_SIZE = POOLED_IMAGE_SIZE * POOLED_IMAGE_SIZE * CONV2_LAYER
FC_SIZE = 1024

OUTPUT_SIZE = 4

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def inference(images):
  # Conv 1
  with tf.name_scope('conv1'):
    weights = tf.Variable(
        tf.truncated_normal([CONV_SIZE, CONV_SIZE, CHANNELS, CONV1_LAYER], stddev=0.01),
        name='weights')
    biases = bias_variable([CONV1_LAYER])
    conv1 = tf.nn.relu(conv2d(images, weights) + biases)
    pool1 = max_pool_2x2(conv1)
  # Conv 2
  with tf.name_scope('conv2'):
    weights = tf.Variable(
        tf.truncated_normal([CONV_SIZE, CONV_SIZE, CONV1_LAYER, CONV2_LAYER], stddev=0.01),
        name='weights')
    biases = bias_variable([CONV2_LAYER])
    conv2 = tf.nn.relu(conv2d(pool1, weights) + biases)
    pool2 = max_pool_2x2(conv2)
  # FC1
  with tf.name_scope('fc1'):
    pool2_flat = tf.reshape(pool2, [-1, POOLED_SIZE])
    weights = tf.Variable(
        tf.truncated_normal([POOLED_SIZE, FC_SIZE],
                            stddev=1.0 / math.sqrt(float(POOLED_SIZE))),
        name='weights')
    biases = bias_variable([FC_SIZE])
    fc = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)
  # FC2
  with tf.name_scope('fc2'):
    weights = tf.Variable(
        tf.truncated_normal([FC_SIZE, OUTPUT_SIZE],
                            stddev=1.0 / math.sqrt(float(FC_SIZE))),
        name='weights')
    biases = bias_variable([OUTPUT_SIZE])
    result = tf.matmul(fc, weights) + biases
  return result

def loss(result, truth):
  return tf.reduce_mean(tf.squared_difference(result, tf.to_float(truth)))

def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
