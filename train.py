from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import face_net

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         face_net.IMAGE_SIZE,
                                                         face_net.IMAGE_SIZE,
                                                         face_net.CHANNELS))
  truth_placeholder = tf.placeholder(tf.int32, shape=(None, 4))
  return images_placeholder, truth_placeholder

def fill_feed_dict(data_set, images_pl, truth_pl):
  images_feed, truth_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      images_pl: images_feed,
      truth_pl: truth_feed,
  }
  return feed_dict

def do_eval(sess,
            evaluation,
            images_placeholder,
            truth_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    truth_placeholder: The truth placeholder.
    data_set: The set of images and truth to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  loss = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               truth_placeholder)
    loss += sess.run(evaluation, feed_dict=feed_dict)
  print('  Num examples: %d  Average loss: %d' %
        (num_examples, loss / steps_per_epoch))

def run_training():
  train_data = input_data.DataSet('train.txt')
  test_data = input_data.DataSet('test.txt')

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and truth.
    images_placeholder, truth_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes result from the inference model.
    result = face_net.inference(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    loss = face_net.loss(result, truth_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = face_net.training(loss, FLAGS.learning_rate)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    total_time = 0
    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and truth
      # for this particular training step.
      feed_dict = fill_feed_dict(train_data,
                                 images_placeholder,
                                 truth_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time
      total_time = total_time + duration
      # Write the summaries and print an overview fairly often.
      if (step+1) % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step+1, loss_value, total_time))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                loss,
                images_placeholder,
                truth_placeholder,
                train_data)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                loss,
                images_placeholder,
                truth_placeholder,
                test_data)

    summary_writer.flush()

    print('Total time: %.3f sec' % (total_time))

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.000001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=50000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='./log/',
      help='Directory to put the log data.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
