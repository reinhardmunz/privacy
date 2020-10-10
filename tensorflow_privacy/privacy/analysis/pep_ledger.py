from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class PepLedger:
  def __init__(self, num_uids):
    self.ledger = tf.Variable(initial_value=tf.zeros((num_uids,),
                                                     tf.float32),
                              trainable=False, name='pep_ledger',
                              use_resource=True)

  def initial_ledger_sample_state(self):
    return tf.SparseTensor(tf.zeros((0, 1), tf.int64), [], self.ledger.shape)

  def record_privacy_loss(self, dense_ledger_sample_state):
    new_ledger = tf.nest.map_structure(tf.add, self.ledger,
                                       dense_ledger_sample_state)
    return self.ledger.assign(new_ledger, use_locking=True, read_value=False)
