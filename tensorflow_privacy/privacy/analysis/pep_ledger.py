from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class PepLedger:
  def __init__(self, num_uids):
    self.ledger = tf.Variable(initial_value=tf.zeros((num_uids,),
                                                     tf.float32),
                              trainable=False, name='pep_internal_ledger',
                              use_resource=True)
    self.min = tf.Variable(initial_value=tf.zeros((), tf.float32),
                           trainable=False, name='pep_internal_ledger_min',
                           use_resource=True)
    self.mean = tf.Variable(initial_value=tf.zeros((), tf.float32),
                            trainable=False, name='pep_internal_ledger_mean',
                            use_resource=True)
    self.max = tf.Variable(initial_value=tf.zeros((), tf.float32),
                           trainable=False, name='pep_internal_ledger_max',
                           use_resource=True)

  def initial_ledger_sample_state(self):
    return tf.SparseTensor(tf.zeros((0, 1), tf.int64), [], self.ledger.shape)

  def record_privacy_loss(self, dense_ledger_sample_state):
    new_ledger = tf.nest.map_structure(tf.add, self.ledger,
                                       dense_ledger_sample_state)
    new_min = tf.reduce_min(new_ledger)
    new_mean = tf.reduce_mean(new_ledger)
    new_max = tf.reduce_max(new_ledger)
    with tf.control_dependencies([
      self.min.assign(new_min, use_locking=True, read_value=False),
      self.mean.assign(new_mean, use_locking=True, read_value=False),
      self.max.assign(new_max, use_locking=True, read_value=False),
    ]):
      return self.ledger.assign(new_ledger, use_locking=True, read_value=False)
