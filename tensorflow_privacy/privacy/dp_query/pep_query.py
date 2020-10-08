from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import dp_query


class PepQuery(dp_query.DPQuery):
  def __init__(self):
    self._ledger = None

  def set_ledger(self, ledger):
    self._ledger = ledger

  def privacy_record_from(self, record):
    return record[0]

  def data_record_from(self, record):
    return record[1]

  def preprocess_record(self, params, record):
    privacy_record = self.privacy_record_from(record)
    data_record = self.data_record_from(record)
    preprocessed_privacy_record = self.preprocess_privacy_record(params,
                                                                 privacy_record)
    preprocessed_data_record = self.preprocess_data_record(params, data_record)
    return preprocessed_privacy_record, preprocessed_data_record

  def preprocess_privacy_record(self, params, privacy_record):
    return 0.0, privacy_record

  def preprocess_data_record(self, params, data_record):
    return data_record

  def privacy_sample_state_from(self, sample_state):
    return sample_state[0]

  def data_sample_state_from(self, sample_state):
    return sample_state[1]

  def ledger_sample_state_from(self, privacy_sample_state):
    return privacy_sample_state

  def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
    privacy_sample_state = self.privacy_sample_state_from(sample_state)
    data_sample_state = self.data_sample_state_from(sample_state)
    preprocessed_privacy_record = self.privacy_record_from(preprocessed_record)
    preprocessed_data_record = self.data_record_from(preprocessed_record)
    new_privacy_sample_state = self.accumulate_preprocessed_privacy_record(
        privacy_sample_state, preprocessed_privacy_record,
        preprocessed_data_record)
    new_data_sample_state = self.accumulate_preprocessed_data_record(
        data_sample_state, preprocessed_data_record)
    return new_privacy_sample_state, new_data_sample_state

  def merge_sample_states(self, sample_state_1, sample_state_2):
    new_privacy_sample_state = self.merge_privacy_sample_states(
        self.privacy_sample_state_from(sample_state_1),
        self.privacy_sample_state_from(sample_state_2))
    new_data_sample_state = self.merge_data_sample_states(
        self.data_sample_state_from(sample_state_1),
        self.data_sample_state_from(sample_state_2))
    return new_privacy_sample_state, new_data_sample_state

  def get_noised_result(self, sample_state, global_state):
    privacy_sample_state = self.privacy_sample_state_from(sample_state)
    data_sample_state = self.data_sample_state_from(sample_state)
    record_op = self.record_privacy_loss(privacy_sample_state)
    data_result, global_state = self.get_noised_data_result(
        privacy_sample_state, data_sample_state, global_state)
    with tf.control_dependencies([record_op]):
      return data_result, global_state

  def record_privacy_loss(self, privacy_sample_state):
    if not self._ledger:
      raise ValueError('CANNOT HAVE NO LEDGER')
    if not privacy_sample_state:
      raise ValueError('privacy_sample_state cannot be None.')
    ledger_sample_state = self.ledger_sample_state_from(privacy_sample_state)
    record_op = self._ledger.record_privacy_loss(ledger_sample_state)
    return record_op

  def initial_sample_state(self, params=None, template=None):
    initial_privacy_sample_state = self.initial_privacy_sample_state(
        params=params, data_template=template)
    initial_data_sample_state = self.initial_data_sample_state(
        data_template=template)
    return initial_privacy_sample_state, initial_data_sample_state

  def merge_privacy_sample_states(self, privacy_sample_state_1,
                                  privacy_sample_state_2):
    if privacy_sample_state_1 is None and privacy_sample_state_2 is None:
      return None
    if privacy_sample_state_1 is None:
      return privacy_sample_state_2
    if privacy_sample_state_2 is None:
      return privacy_sample_state_1
    return tf.add(privacy_sample_state_1, privacy_sample_state_2)

  def initial_privacy_sample_state(self, params=None, data_template=None):
    return None

  @abc.abstractmethod
  def accumulate_preprocessed_privacy_record(self, privacy_sample_state,
                                             preprocessed_privacy_record,
                                             preprocessed_data_record):
    pass

  @abc.abstractmethod
  def accumulate_preprocessed_data_record(self, data_sample_state,
                                          preprocessed_data_record):
    pass

  @abc.abstractmethod
  def merge_data_sample_states(self, data_sample_state_1, data_sample_state_2):
    pass

  @abc.abstractmethod
  def initial_data_sample_state(self, data_template=None):
    pass

  @abc.abstractmethod
  def get_noised_data_result(self, privacy_sample_state, data_sample_state,
                             global_state):
    pass


class SumAggregationPepQuery(PepQuery):
  def accumulate_preprocessed_privacy_record(self, privacy_sample_state,
                                             preprocessed_privacy_record,
                                             preprocessed_data_record):
    return None

  def get_noised_data_result(self, privacy_sample_state, data_sample_state,
                             global_state):
    return data_sample_state, global_state

  def accumulate_preprocessed_data_record(self, data_sample_state,
                                          preprocessed_data_record):
    return tf.nest.map_structure(dp_query.safe_add, data_sample_state,
                                 preprocessed_data_record)

  def merge_data_sample_states(self, data_sample_state_1, data_sample_state_2):
    return tf.nest.map_structure(tf.add, data_sample_state_1,
                                 data_sample_state_2)

  def initial_data_sample_state(self, data_template=None):
    return tf.nest.map_structure(dp_query.zeros_like, data_template)
