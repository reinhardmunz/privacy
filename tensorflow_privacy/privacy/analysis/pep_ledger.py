from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


class PepLedger:
  def __init__(self, backing_array):
    self.backing_array = backing_array
    self.ledger = None

  def initialize(self):
    self.ledger = tf.Variable(
        initial_value=tf.convert_to_tensor(
            self.backing_array, dtype=tf.float32),
        trainable=False, name='ledger', use_resource=True)
    return self

  def initial_ledger_sample_state(self):
    return tf.SparseTensor(tf.zeros((0, 1), tf.int64), [], self.ledger.shape)

  def record_privacy_loss(self, ledger_sample_state):
    #dense_sample = tf.sparse.to_dense(ledger_sample_state)
    dense_sample = tf.constant(np.ones(60000))
    return self.ledger.assign_add(dense_sample,
                                  use_locking=True,
                                  name='record_privacy_loss',
                                  read_value=False)
