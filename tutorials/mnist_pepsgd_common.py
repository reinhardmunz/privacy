from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def make_input_fn(split, input_batch_size=256, repetitions=-1, tpu=False):
  """Make input function on given MNIST split."""

  def input_fn(params=None):
    """A simple input function."""
    batch_size = params.get('batch_size', input_batch_size)

    def parser(uid, example):
      image, label = example['image'], example['label']
      uid = tf.cast(uid, tf.int32)
      image = tf.cast(image, tf.float32)
      image /= 255.0
      label = tf.cast(label, tf.int32)
      metadata = tf.stack([uid, label])
      return image, metadata

    dataset = tfds.load(name='mnist', split=split)
    dataset = dataset.enumerate().map(parser).shuffle(60000).\
        repeat(repetitions).batch(batch_size)
    # If this input function is not meant for TPUs, we can stop here.
    # Otherwise, we need to explicitly set its shape. Note that for unknown
    # reasons, returning the latter format causes performance regression
    # on non-TPUs.
    if not tpu:
      return dataset

    # Give inputs statically known shapes; needed for TPUs.
    images, metadata = tf.data.make_one_shot_iterator(dataset).get_next()
    # return images, metadata
    images.set_shape([batch_size, 28, 28, 1])
    metadata.set_shape([batch_size, 2])
    return images, metadata

  return input_fn
