from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import distutils

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import pep_query


class PepGaussianSumQuery(pep_query.SumAggregationPepQuery):
  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip', 'stddev'])

  def __init__(self, l2_norm_clip, stddev, ledger=None):
    super().__init__()
    if stddev <= 0:
      raise ValueError("PepGaussianQuery: stddev must be positive")
    self._l2_norm_clip = l2_norm_clip
    self._stddev = stddev
    if ledger is not None:
      self.set_ledger(ledger)

  def make_global_state(self, l2_norm_clip, stddev):
    """Creates a global state from the given parameters."""
    return self._GlobalState(tf.cast(l2_norm_clip, tf.float32),
                             tf.cast(stddev, tf.float32))

  def initial_global_state(self):
    return self.make_global_state(self._l2_norm_clip, self._stddev)

  def derive_sample_params(self, global_state):
    return global_state.l2_norm_clip, global_state.stddev

  def derive_initial_sample_params(self, global_state):
    return global_state.stddev

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
    preprocessed_data_record, _ = self.preprocess_data_record_impl(params[0],
                                                                   data_record)
    return preprocessed_data_record

  def preprocess_privacy_record(self, params, privacy_record):
    return params[1], privacy_record

  def initial_privacy_sample_state(self, params=None, data_template=None):
    if params is None:
      raise ValueError("PepGaussianQuery: params may not be None")
    if data_template is None:
      raise ValueError("PepGaussianQuery: data_template may not be None")
    structure = tf.nest.map_structure(dp_query.zeros_like, data_template)

    if distutils.version.LooseVersion(
        tf.__version__) < distutils.version.LooseVersion('2.0.0'):

      def add_noise(v):
        return v + tf.random.normal(
            tf.shape(input=v), stddev=params, dtype=v.dtype)
    else:
      random_normal = tf.random_normal_initializer(
          stddev=params)

      def add_noise(v):
        return v + tf.cast(random_normal(tf.shape(input=v)), dtype=v.dtype)

    noise = tf.nest.map_structure(add_noise, structure)
    initial_ledger_sample_state = self._ledger.initial_ledger_sample_state()
    return noise, initial_ledger_sample_state

  def noise_from(self, privacy_sample_state):
    return privacy_sample_state[0]

  def ledger_sample_state_from(self, privacy_sample_state):
    return privacy_sample_state[1]

  def calculate_privacy_loss_for(self, noise, stddev, data_record):
    def special_reshape(v):
      return tf.reshape(v, [-1])

    flat_data_lst = tf.nest.map_structure(special_reshape, data_record)
    flat_data = tf.concat(flat_data_lst, axis=0)
    true_dist = tfp.distributions.Normal(loc=flat_data, scale=stddev)
    flat_zeros = dp_query.zeros_like(flat_data)
    zero_dist = tfp.distributions.Normal(loc=flat_zeros, scale=stddev)
    flat_noise_lst = tf.nest.map_structure(special_reshape, noise)
    flat_noise = tf.concat(flat_noise_lst, axis=0)
    flat_noised_result = tf.nest.map_structure(tf.add, flat_data, flat_noise)
    true_log_prob = true_dist.log_prob(flat_noised_result)
    zero_log_prob = zero_dist.log_prob(flat_noised_result)
    privacy_losses = tf.nest.map_structure(tf.subtract, true_log_prob,
                                           zero_log_prob)
    return tf.reduce_sum(privacy_losses)

  def accumulate_preprocessed_privacy_record(self, privacy_sample_state,
                                             preprocessed_privacy_record,
                                             preprocessed_data_record):
    noise = self.noise_from(privacy_sample_state)
    ledger_sample_state = self.ledger_sample_state_from(privacy_sample_state)
    stddev = preprocessed_privacy_record[0]
    uid = preprocessed_privacy_record[1]
    privacy_loss = self.calculate_privacy_loss_for(noise, stddev,
                                                   preprocessed_data_record)
    ledger_entry = tf.SparseTensor([[uid]], values=[privacy_loss],
                                   dense_shape=ledger_sample_state.dense_shape)
    new_ledger_sample_state = tf.nest.map_structure(tf.sparse_add,
                                                    ledger_sample_state,
                                                    ledger_entry)
    return noise, new_ledger_sample_state

  def get_noised_data_result(self, privacy_sample_state, data_sample_state,
                             global_state):
    noise = self.noise_from(privacy_sample_state)
    noised_data_result = tf.nest.map_structure(tf.add, data_sample_state, noise)
    record_op = self.record_privacy_loss(privacy_sample_state)
    with tf.control_dependencies([record_op]):
      return noised_data_result, global_state
