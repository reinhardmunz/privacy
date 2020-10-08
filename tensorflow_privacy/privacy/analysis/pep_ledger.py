from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class PepLedger:
  def __init__(self, num_uids):
    self.num_uids = tf.constant(num_uids, dtype=tf.int64)
    self.ledger = tf.Variable(initial_value=tf.zeros([num_uids]),
                              trainable=False, name='ledger')
    self._loss_count = tf.Variable(initial_value=0, trainable=False,
                                   name='loss_count', dtype=tf.int64)
    self._cs = tf.CriticalSection()

  def initial_ledger_sample_state(self):
    shape = tf.shape(self.ledger, out_type=tf.dtypes.int64)
    return tf.SparseTensor(tf.zeros((0, 1), tf.int64), [], shape)

  def record_privacy_loss(self, ledger_sample_state):
    def _do_record_privacy_loss():
      dense_sample = tf.sparse.to_dense(ledger_sample_state)
      num_losses = tf.count_nonzero(dense_sample)

      def _do_record():
        return self.ledger.assign_add(dense_sample, read_value=False)

      def _do_print_record():
        self.print_stats()
        return _do_record()

      with tf.control_dependencies(
            [self._loss_count.assign_add(num_losses, read_value=False)]):
        return tf.cond(tf.equal(tf.mod(self._loss_count, self.num_uids),
                                tf.constant(0, dtype=tf.int64)),
                       _do_print_record,
                       _do_record)

    return self._cs.execute(_do_record_privacy_loss)

  def print_stats(self):
    non_zero_loss = tf.count_nonzero(self.ledger)
    min_loss = tf.reduce_min(self.ledger)
    max_loss = tf.reduce_max(self.ledger)
    mean_loss = tf.reduce_mean(self.ledger)
    median_loss = tfp.stats.percentile(self.ledger, 50)
    logging.info("PepLedger STATS: min=%s median=%s max=%s mean=%s non_zero=%s",
                 min_loss, median_loss, max_loss, mean_loss, non_zero_loss)
