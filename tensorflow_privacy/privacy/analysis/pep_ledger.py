from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class PepLedger:
  def __init__(self, backing_array):
    self.backing_array = backing_array
    self.ledger = None
    self._cs = None

  def initialize(self):
    self.ledger = tf.Variable(
        initial_value=tf.convert_to_tensor(
            self.backing_array, dtype=tf.float32),
        trainable=False, name='ledger')
    self._cs = tf.CriticalSection()
    return self

  def initial_ledger_sample_state(self):
    return tf.SparseTensor(tf.zeros((0, 1), tf.int64), [], self.ledger.shape)

  def record_privacy_loss(self, ledger_sample_state):
    def _do_record_privacy_loss():
      dense_sample = tf.sparse.to_dense(ledger_sample_state)
      return self.ledger.assign_add(dense_sample, read_value=False)
    return self._cs.execute(_do_record_privacy_loss)
