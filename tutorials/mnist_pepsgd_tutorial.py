"""Train a CNN on MNIST with differentially private PEP-SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from tensorflow.python.ops import array_ops

from tensorflow_privacy.privacy.analysis import pep_ledger
from tensorflow_privacy.privacy.optimizers import pep_optimizer

import mnist_dpsgd_tutorial_common as dp_common
import mnist_pepsgd_common as pep_common

import numpy as np

flags.DEFINE_boolean(
    'pepsgd', True, 'If True, train with PEP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 30, 'Number of epochs')
flags.DEFINE_string('model_dir', '/tmp/munz-tf-model', 'Model directory')

FLAGS = flags.FLAGS


def cnn_model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
  """Model function for a CNN."""

  # Define CNN architecture.
  logits = dp_common.get_cnn_model(features)

  uids, real_labels = tf.split(labels, 2, axis=1)
  uids = tf.squeeze(uids)
  real_labels = tf.squeeze(real_labels)

  uids = array_ops.stop_gradient(uids)

  # Calculate loss as a vector.
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=real_labels, logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_global_step()
    if FLAGS.pepsgd:
      # Use PEP version of GradientDescentOptimizer. Other optimizers are
      # available in dp_optimizer. Most optimizers inheriting from
      # tf.train.Optimizer should be wrappable in differentially private
      # counterparts by calling dp_optimizer.optimizer_from_args().
      ledger = pep_ledger.PepLedger(60000)
      optimizer = pep_optimizer.PepGradientDescentGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          ledger=ledger,
          learning_rate=FLAGS.learning_rate)
      train_op = optimizer.minimize(loss=vector_loss, global_step=global_step,
                                    uids=uids)
      tf.summary.scalar("min_priv_loss", ledger.min)
      tf.summary.scalar("mean_priv_loss", ledger.mean)
      tf.summary.scalar("max_priv_loss", ledger.max)
    else:
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=FLAGS.learning_rate)
      train_op = optimizer.minimize(loss=scalar_loss, global_step=global_step)

    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=real_labels,
                predictions=tf.argmax(input=logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  logging.set_verbosity(logging.INFO)

  # Instantiate the tf.Estimator.
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                            model_dir=FLAGS.model_dir)

  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    # Train the model for one epoch.
    mnist_classifier.train(
        input_fn=pep_common.make_input_fn('train', FLAGS.batch_size),
        steps=steps_per_epoch)
    end_time = time.time()
    logging.info('Epoch %d time in seconds: %.2f', epoch, end_time - start_time)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        input_fn=pep_common.make_input_fn('test', FLAGS.batch_size, 1))
    test_accuracy = eval_results['accuracy']
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

    if FLAGS.model_dir is not None:
      current_global_step = tf.train.load_variable(FLAGS.model_dir,
                                                   'global_step')
      current_ledger = tf.train.load_variable(FLAGS.model_dir,
                                              'pep_internal_ledger')
      print(f"Ledger privacy loss stats after {current_global_step} steps are: "
            f"min_priv_loss={np.min(current_ledger):.3f} "
            f"median_priv_loss={np.median(current_ledger):.3f} "
            f"mean_priv_loss={np.mean(current_ledger):.3f} "
            f"max_priv_loss={np.max(current_ledger):.3f}")

    if not FLAGS.pepsgd:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
