from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import distutils

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from absl import logging

from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import pep_query


class PepGaussianSumQuery(pep_query.SumAggregationPepQuery):
  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip'])

  def __init__(self, l2_norm_clip, noise, ledger=None):
    super().__init__()
    self._l2_norm_clip = l2_norm_clip
    self._noise = noise
    if ledger is not None:
      self.set_ledger(ledger)

  def make_global_state(self, l2_norm_clip):
    """Creates a global state from the given parameters."""
    return self._GlobalState(tf.cast(l2_norm_clip, tf.float32))

  def initial_global_state(self):
    return self.make_global_state(self._l2_norm_clip)

  def derive_sample_params(self, global_state):
    return global_state.l2_norm_clip

  def preprocess_data_record_impl(self, params, data_record):
    """Clips the l2 norm, returning the clipped record and the l2 norm.

    Args:
      params: The parameters for the sample.
      data_record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
        the structure of preprocessed tensors, and l2_norm is the total l2 norm
        before clipping.
    """
    l2_norm_clip = params
    record_as_list = tf.nest.flatten(data_record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    return tf.nest.pack_sequence_as(data_record, clipped_as_list), norm

  def preprocess_data_record(self, params, data_record):
    preprocessed_data_record, _ = self.preprocess_data_record_impl(params,
                                                                   data_record)
    return preprocessed_data_record

  def initial_privacy_sample_state(self, params=None, data_template=None):
    return self._ledger.initial_ledger_sample_state()

  def calculate_privacy_loss_for(self, data_record):
    def special_reshape(v):
      return tf.reshape(v, [-1])

    flat_data_lst = tf.nest.map_structure(special_reshape, data_record)
    flat_data = tf.squeeze(tf.concat(flat_data_lst, axis=0))
    noise = self._noise.get_noise(flat_data)
    with tf.control_dependencies(tf.nest.flatten(noise)):
      flat_data_mult = tf.expand_dims(flat_data, axis=0)
      flat_data_mult = tf.repeat(flat_data_mult, self._noise.steps_per_epoch,
                                 axis=0)
      true_dist = tfp.distributions.Normal(loc=flat_data_mult,
                                           scale=self._noise.stddev)
      flat_zeros = dp_query.zeros_like(flat_data_mult)
      zero_dist = tfp.distributions.Normal(loc=flat_zeros,
                                           scale=self._noise.stddev)
      step_in_epoch = tf.cast(self._noise.get_step_in_epoch(), dtype=tf.int32)
      behind = tf.constant(self._noise.steps_per_epoch - 1)
      behind = tf.subtract(behind, step_in_epoch)
      padding = tf.stack([step_in_epoch, behind], 0)
      padding = tf.expand_dims(padding, axis=0)
      padded_data = tf.pad(flat_data, padding)
      flat_noised_result = tf.nest.map_structure(tf.add, padded_data, noise)
      true_log_prob = true_dist.log_prob(flat_noised_result)
      zero_log_prob = zero_dist.log_prob(flat_noised_result)
      privacy_losses = tf.nest.map_structure(tf.subtract, true_log_prob,
                                             zero_log_prob)
      per_batch_losses = tf.reduce_sum(privacy_losses, 1)
    return tfp.math.reduce_logmeanexp(per_batch_losses)

  def accumulate_preprocessed_privacy_record(self, privacy_sample_state,
                                             preprocessed_privacy_record,
                                             preprocessed_data_record):
    privacy_loss = self.calculate_privacy_loss_for(preprocessed_data_record)
    ledger_entry = tf.SparseTensor([[preprocessed_privacy_record]],
                                   values=[privacy_loss],
                                   dense_shape=privacy_sample_state.dense_shape)
    new_ledger_sample_state = tf.nest.map_structure(tf.sparse_add,
                                                    privacy_sample_state,
                                                    ledger_entry)
    return new_ledger_sample_state

  def get_noised_data_result(self, privacy_sample_state, data_sample_state,
                             global_state):
    dense_ledger_sample_state = self.dense_from(privacy_sample_state)
    with tf.control_dependencies(tf.nest.flatten(dense_ledger_sample_state)):
      noise = self._noise.get_batch_noise()

      def special_reshape(v):
        return tf.reshape(v, [-1])

      flat_data_lst = tf.nest.map_structure(special_reshape, data_sample_state)
      flat_data = tf.squeeze(tf.concat(flat_data_lst, axis=0))
      noised_data = tf.nest.map_structure(tf.add, flat_data, noise)
      split_sizes = tf.nest.map_structure(lambda x: x.shape[0], flat_data_lst)
      splits = tf.split(noised_data, split_sizes)

      def special_merge(s, d):
        return tf.reshape(s, d.shape)

      result_data = tf.nest.map_structure(special_merge, splits,
                                          data_sample_state)
      record_op = self.record_privacy_loss(dense_ledger_sample_state)
    with tf.control_dependencies([record_op]):
      final_result = tf.nest.map_structure(tf.identity, result_data)
      return final_result, global_state


class PepGaussianNoise:
  def __init__(self, global_step, steps_per_epoch, stddev):
    self.global_step = global_step
    self.steps_per_epoch = steps_per_epoch
    if stddev <= 0:
      raise ValueError("PepGaussianQuery: stddev must be positive")
    self.stddev = tf.cast(stddev, tf.float32)
    self.noise = tf.Variable(initial_value=tf.zeros(()), trainable=False,
                             validate_shape=False, name='noise',
                             shape=tf.TensorShape(None), use_resource=True)

  def get_noise(self, data_template):
    def assign_noise():
      if data_template is None:
        raise ValueError("PepGaussianNoise: data_template may not be None")
      zeros = tf.zeros(data_template.shape, dtype=tf.float32)
      zeros = tf.expand_dims(zeros, axis=0)
      zeros = tf.repeat(zeros, self.steps_per_epoch, axis=0)
      if distutils.version.LooseVersion(
          tf.__version__) < distutils.version.LooseVersion('2.0.0'):
        def add_noise(v):
          return v + tf.random.normal(
            tf.shape(input=v), stddev=self.stddev, dtype=v.dtype)
      else:
        random_normal = tf.random_normal_initializer(stddev=self.stddev)
        def add_noise(v):
          return v + tf.cast(random_normal(tf.shape(input=v)), dtype=v.dtype)
      noise = tf.nest.map_structure(add_noise, zeros)
      with tf.control_dependencies(tf.nest.flatten(noise)):
        return self.noise.assign(noise, use_locking=True, read_value=False)

    step_in_epoch = tf.mod(self.global_step.read_value(), self.steps_per_epoch)
    op = tf.cond(tf.equal(step_in_epoch, tf.zeros((), dtype=tf.int64)),
                 true_fn=assign_noise, false_fn=lambda: tf.no_op())
    with tf.control_dependencies([op]):
      return self.noise.read_value()

  def get_step_in_epoch(self):
    return tf.mod(self.global_step.read_value(), self.steps_per_epoch)

  def get_batch_noise(self):
    return tf.squeeze(
      tf.slice(self.noise.read_value(), [self.get_step_in_epoch(), 0], [1, -1]))
