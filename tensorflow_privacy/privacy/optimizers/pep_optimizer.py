"""Pep-Differentially private optimizers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import pep_gaussian_query


def make_optimizer_class(cls):
  """Constructs a Pep optimizer class from an existing one."""
  GATE_OP = tf.train.Optimizer.GATE_OP  # pylint: disable=invalid-name

  cg_parent_code = tf.train.Optimizer.compute_gradients.__code__
  ag_parent_code = tf.train.Optimizer.apply_gradients.__code__
  m_parent_code = tf.train.Optimizer.minimize.__code__

  has_compute_gradients = hasattr(cls, 'compute_gradients')
  if has_compute_gradients:
    cg_child_code = cls.compute_gradients.__code__
  has_apply_gradients = hasattr(cls, 'apply_gradients')
  if has_apply_gradients:
    ag_child_code = cls.apply_gradients.__code__
  has_minimize = hasattr(cls, 'minimize')
  if has_minimize:
    m_child_code = cls.minimize.__code__

  if has_compute_gradients and cg_child_code is not cg_parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)
  if has_apply_gradients and ag_child_code is not ag_parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method apply_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)
  if has_minimize and m_child_code is not m_parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method minimize(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class PepOptimizerClass(cls):
    """Pep-Differentially private subclass of given class cls."""
    def __init__(
        self,
        pep_sum_query,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the PepOptimizerClass.

      Args:
        pep_sum_query: PepQuery object, specifying differential privacy
          mechanism to use.
      """
      super(PepOptimizerClass, self).__init__(*args, **kwargs)
      self._pep_sum_query = pep_sum_query
      self._global_state = self._pep_sum_query.initial_global_state()
      self._was_compute_gradients_called = False
      logging.info("Finished init of special optimizer")

    def minimize(self, loss, uids, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
      grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, uids=uids, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

      vars_with_grad = [v for g, v in grads_and_vars if g is not None]
      if not vars_with_grad:
        raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

      return self.apply_gradients(grads_and_vars, global_step=global_step,
                                  name=name)

    def compute_gradients(self,
                          loss,
                          var_list,
                          uids,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None):
      logging.info("Entered compute_gradients")
      self._was_compute_gradients_called = True
      if callable(loss):
        # TF is running in Eager mode, check we received a vanilla tape.
        logging.info("compute_gradients: in eager mode")
        if not gradient_tape:
          raise ValueError('When in Eager mode, a tape needs to be passed.')

        vector_loss = loss()
        initial_sample_params = (
            self._pep_sum_query.derive_initial_sample_params(
                self._global_state))
        sample_state = self._pep_sum_query.initial_sample_state(
            params=initial_sample_params, template=var_list)
        sample_params = (
            self._pep_sum_query.derive_sample_params(self._global_state))

        def process_single_loss(i, sample_state):
          """Process one loss value (record) with privacy helper."""
          single_loss = tf.gather(vector_loss, i)
          single_uid = tf.gather(uids, i)
          with gradient_tape.stop_recording():
            grads = gradient_tape.gradient(single_loss, var_list)
          sample_state = self._pep_sum_query.accumulate_record(
              sample_params, sample_state, tf.tuple(single_uid, grads))
          return sample_state

        if tf.shape(input=vector_loss)[0] != tf.shape(input=uids)[0]:
          raise ValueError('Sizes of loss and uids do not match.')

        for idx in range(tf.shape(input=vector_loss)[0]):
          sample_state = process_single_loss(idx, sample_state)

        grad_sums, self._global_state = (
            self._pep_sum_query.get_noised_result(
                sample_state, self._global_state))

        def normalize(v):
          return v / tf.cast(tf.shape(input=vector_loss)[0], tf.float32)

        final_grads = tf.nest.map_structure(normalize, grad_sums)

        grads_and_vars = list(zip(final_grads, var_list))
        logging.info("Leaving compute_gradients")
        return grads_and_vars

      else:
        # TF is running in graph mode, check we did not receive a gradient tape.
        logging.info("compute_gradients: in graph mode")

        if gradient_tape:
          raise ValueError('When in graph mode, a tape should not be passed.')

        initial_sample_params = (
            self._pep_sum_query.derive_initial_sample_params(
                self._global_state))
        sample_params = (
            self._pep_sum_query.derive_sample_params(self._global_state))

        def process_single_loss(i, sample_state):
          """Process one microbatch (record) with privacy helper."""
          self_super = super(PepOptimizerClass, self)

          single_loss = tf.squeeze(tf.gather(loss, [i]))
          single_uid = tf.squeeze(tf.gather(uids, [i]))

          if hasattr(self_super, 'compute_gradients'):
            # This case covers optimizers in tf.train.
            compute_gradients_fn = self_super.compute_gradients
          else:
            # This case covers Keras optimizers from optimizers_v2.
            compute_gradients_fn = self_super._compute_gradients  # pylint: disable=protected-access

          grads, _ = zip(*compute_gradients_fn(
              single_loss, var_list, gate_gradients,
              aggregation_method, colocate_gradients_with_ops, grad_loss))
          grads_list = list(grads)

          sample_state = self._pep_sum_query.accumulate_record(
              sample_params, sample_state, (single_uid, grads_list))

          return sample_state

        if var_list is None:
          var_list = (
              tf.trainable_variables() + tf.get_collection(
                  tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        sample_state = self._pep_sum_query.initial_sample_state(
            params=initial_sample_params, template=var_list)

        # Use of while_loop here requires that sample_state be a nested
        # structure of tensors. In general, we would prefer to allow it to be
        # an arbitrary opaque type.
        cond_fn = lambda i, _: tf.less(i, tf.shape(input=loss)[0])
        body_fn = lambda i, state: [tf.add(i, 1), process_single_loss(i, state)]
        idx = tf.constant(0)
        _, sample_state = tf.while_loop(cond=cond_fn, body=body_fn,
                                        loop_vars=[idx, sample_state],
                                        parallel_iterations=48)

        grad_sums, self._global_state = (
            self._pep_sum_query.get_noised_result(
                sample_state, self._global_state))

        def normalize(v):
          try:
            return tf.truediv(v, tf.cast(tf.shape(input=loss)[0], tf.float32))
          except TypeError:
            return None

        final_grads = tf.nest.map_structure(normalize, grad_sums)
        grads_and_vars = list(zip(final_grads, var_list))
        logging.info("Leaving compute_gradients")
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      assert self._was_compute_gradients_called, (
          'compute_gradients() on the differentially private optimizer was not'
          ' called. Which means that the training is not differentially '
          'private. It happens for example in Keras training in TensorFlow '
          '2.0+.')
      logging.info("Applying_gradients")
      return super(PepOptimizerClass,
                   self).apply_gradients(grads_and_vars, global_step, name)

  return PepOptimizerClass


def make_gaussian_optimizer_class(cls):
  """Constructs a Pep optimizer with Gaussian averaging of updates."""

  class PepGaussianOptimizerClass(make_optimizer_class(cls)):
    """Pep subclass of given class cls using Gaussian averaging."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        ledger=None,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._base_optimizer_class = cls

      pep_sum_query = pep_gaussian_query.PepGaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier, ledger=ledger)

      super(PepGaussianOptimizerClass, self).__init__(
          pep_sum_query,
          *args,
          **kwargs)

    def get_config(self):
      """Creates configuration for Keras serialization.

      This method will be called when Keras creates model checkpoints
      and is necessary so that deserialization can be performed.

      Returns:
        A dict object storing arguments to be passed to the __init__ method
        upon deserialization.
      """

      config = self._base_optimizer_class.get_config(self)
      config.update({
          'l2_norm_clip': self._l2_norm_clip,
          'noise_multiplier': self._noise_multiplier})

      raise ValueError('BIG BAD ERROR')

      return config

    @property
    def ledger(self):
      return self._pep_sum_query.ledger

  return PepGaussianOptimizerClass

AdagradOptimizer = tf.train.AdagradOptimizer
AdamOptimizer = tf.train.AdamOptimizer
GradientDescentOptimizer = tf.train.GradientDescentOptimizer
RMSPropOptimizer = tf.train.RMSPropOptimizer

PepAdagradOptimizer = make_optimizer_class(AdagradOptimizer)
PepAdamOptimizer = make_optimizer_class(AdamOptimizer)
PepGradientDescentOptimizer = make_optimizer_class(GradientDescentOptimizer)
PepRMSPropOptimizer = make_optimizer_class(RMSPropOptimizer)

PepAdagradGaussianOptimizer = make_gaussian_optimizer_class(AdagradOptimizer)
PepAdamGaussianOptimizer = make_gaussian_optimizer_class(AdamOptimizer)
PepGradientDescentGaussianOptimizer = make_gaussian_optimizer_class(
    GradientDescentOptimizer)
PepRMSPropGaussianOptimizer = make_gaussian_optimizer_class(RMSPropOptimizer)
