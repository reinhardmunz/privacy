"""Training a language model (recurrent neural network) with PEP-SGD optimizer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import time

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from tensorflow.python.summary import summary
from tensorflow.python.ops import array_ops

from tensorflow_privacy.privacy.dp_query import pep_gaussian_query
from tensorflow_privacy.privacy.analysis import pep_ledger
from tensorflow_privacy.privacy.optimizers import pep_optimizer

import numpy as np

flags.DEFINE_boolean(
    'pepsgd', True, 'If True, train with PEP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.001,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('num_train_samples', 45000, 'Number of training samples')
flags.DEFINE_integer('num_test_samples', 10000, 'Number of testing samples')
flags.DEFINE_integer('batch_size', 250, 'Batch size '
                     '(must evenly divide num_train_samples/num_test_samples)')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_string('model_dir', '/tmp/munz-tf-model', 'Model directory')

FLAGS = flags.FLAGS

SEQ_LEN = 80


def rnn_model_fn(features, labels, mode):  # pylint: disable=unused-argument
  """Model function for a RNN."""

  # Define RNN architecture using tf.keras.layers.
  x = features['x']
  x = tf.reshape(x, [-1, SEQ_LEN])
  input_layer = x[:, :-1]
  input_one_hot = tf.one_hot(input_layer, 256)
  lstm = tf.keras.layers.LSTM(256, return_sequences=True).apply(input_one_hot)
  logits = tf.keras.layers.Dense(256).apply(lstm)

  uids = array_ops.stop_gradient(labels)

  # Calculate loss as a vector.
  vector_loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=tf.cast(tf.one_hot(x[:, 1:], 256), dtype=tf.float32),
      logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_global_step()
    if FLAGS.pepsgd:
      ledger = pep_ledger.PepLedger(FLAGS.num_train_samples)
      steps_per_epoch = FLAGS.num_train_samples // FLAGS.batch_size
      stddev = FLAGS.l2_norm_clip * FLAGS.noise_multiplier
      noise = pep_gaussian_query.PepGaussianNoise(global_step, steps_per_epoch,
                                                  stddev)

      optimizer = pep_optimizer.PepAdamGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise=noise,
          ledger=ledger,
          learning_rate=FLAGS.learning_rate)
      train_op = optimizer.minimize(loss=vector_loss, uids=uids,
                                    global_step=global_step)
      summary.histogram("priv_loss", ledger.ledger)
      summary.scalar("priv_loss_min", tf.reduce_min(ledger.ledger))
      summary.scalar("priv_loss_mean", tf.reduce_mean(ledger.ledger))
      summary.scalar("priv_loss_max", tf.reduce_max(ledger.ledger))
      summary.histogram("noise", noise.noise)
      summary.scalar("noise_min", tf.reduce_min(noise.noise))
      summary.scalar("noise_mean", tf.reduce_mean(noise.noise))
      summary.scalar("noise_max", tf.reduce_max(noise.noise))
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=scalar_loss, train_op=train_op)
    else:
      optimizer = tf.train.AdamOptimizer(
          learning_rate=FLAGS.learning_rate)
      train_op = optimizer.minimize(loss=scalar_loss, global_step=global_step)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=tf.cast(x[:, 1:], dtype=tf.int32),
                predictions=tf.argmax(input=logits, axis=2))
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)

def make_input_fn(split, input_size=45000, input_batch_size=250, repetitions=-1,
                  tpu=False):
  """Make input function on given MNIST split."""

  def input_fn(params=None):
    """A simple input function."""
    batch_size = params.get('batch_size', input_batch_size)

    def parser(uid, example):
      data = example['text'].flatten()
      uid = tf.cast(uid, tf.int32)
      return data, uid

    dataset = tfds.load(name='lm1b/subwords8k', split=split,
                        shuffle_files=False)
    dataset = dataset.take(input_size).enumerate().map(parser).\
        shuffle(input_size).repeat(repetitions).batch(batch_size)

    dataset = dataset.enumerate().map(parser).shuffle(60000).\
        repeat(repetitions).batch(batch_size)
    # If this input function is not meant for TPUs, we can stop here.
    # Otherwise, we need to explicitly set its shape. Note that for unknown
    # reasons, returning the latter format causes performance regression
    # on non-TPUs.
    if not tpu:
      return dataset

    raise RuntimeError("this input function does not support tpu operation")

  return input_fn


def main(unused_argv):
  assert FLAGS.num_train_samples % FLAGS.batch_size == 0
  assert FLAGS.num_test_samples % FLAGS.batch_size == 0
  assert not path.exists(FLAGS.model_dir)

  logging.set_verbosity(logging.INFO)

  steps_per_epoch = FLAGS.num_train_samples // FLAGS.batch_size

  # Instantiate the tf.Estimator.
  run_config = tf.estimator.RunConfig(save_summary_steps=1,
                                      log_step_count_steps=1,
                                      keep_checkpoint_max=None)
  lm_classifier = tf.estimator.Estimator(model_fn=rnn_model_fn,
                                         model_dir=FLAGS.model_dir,
                                         config=run_config)

  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    # Train the model for one epoch.
    lm_classifier.train(input_fn=make_input_fn('train', FLAGS.num_train_samples,
                                               FLAGS.batch_size),
                        steps=steps_per_epoch)
    end_time = time.time()
    logging.info('Epoch %d time in seconds: %.2f', epoch, end_time - start_time)

    if epoch % 5 == 0:
      eval_results = lm_classifier.evaluate(input_fn=
                                            make_input_fn(
                                              'test',
                                              FLAGS.num_test_samples,
                                              FLAGS.batch_size))
      test_accuracy = eval_results['accuracy']
      print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

    if FLAGS.model_dir is not None:
      current_global_step = tf.train.load_variable(FLAGS.model_dir,
                                                   'global_step')
      current_ledger = tf.train.load_variable(FLAGS.model_dir,
                                              'pep_internal_ledger')
      current_noise = tf.train.load_variable(FLAGS.model_dir,
                                             'pep_internal_noise')
      print(f"Ledger privacy loss stats after {current_global_step} steps are: "
            f"priv_loss_min={np.min(current_ledger):.3f} "
            f"priv_loss_median={np.median(current_ledger):.3f} "
            f"priv_loss_mean={np.mean(current_ledger):.3f} "
            f"priv_loss_max={np.max(current_ledger):.3f}")
      print(f"Noise stats after {current_global_step} steps are: "
            f"noise_min={np.min(current_noise):.3f} "
            f"noise_median={np.median(current_noise):.3f} "
            f"noise_mean={np.mean(current_noise):.3f} "
            f"noise_max={np.max(current_noise):.3f}")

    if not FLAGS.pepsgd:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
