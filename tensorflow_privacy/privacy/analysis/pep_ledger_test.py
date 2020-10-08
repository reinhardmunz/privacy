from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import pep_ledger
from tensorflow_privacy.privacy.dp_query import pep_gaussian_query
from tensorflow_privacy.privacy.dp_query import pep_test_utils


tf.enable_eager_execution()


class PepLedgerTest(tf.test.TestCase):
  def test_basic(self):
    ledger = pep_ledger.PepLedger(5)
    ledger.record_privacy_loss(
        tf.SparseTensor([[1], [3], [4]], [1, 1.5, 4], [5]))
    ledger.record_privacy_loss(tf.SparseTensor([[2], [3]], [2, 1.5], [5]))
    self.assertAllClose(ledger.ledger, tf.constant([0, 1, 2, 3, 4]))

  def test_pep_gaussian_query(self):
    record1 = (0, tf.constant([1.0, 2.0, 3.0]))
    record2 = (2, tf.constant([4.0, 5.0, 6.0]))
    ledger = pep_ledger.PepLedger(5)
    query = pep_gaussian_query.PepGaussianQuery(100.0, 0.01, ledger)
    result, _ = pep_test_utils.run_query(query, [record1, record2])
    loss_greater_equal = tf.nest.map_structure(
        tf.greater_equal, ledger.ledger,
        tf.constant([1000.0, 0.0, 1000.0, 0.0, 0.0]))
    self.assertTrue(tf.reduce_all(loss_greater_equal))
    self.assertAllClose(result, tf.constant([5.0, 7.0, 9.0]), atol=0.1)

if __name__ == '__main__':
  tf.test.main()
