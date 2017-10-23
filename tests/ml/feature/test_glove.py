import numpy as np
import unittest

import cocoscore.ml.feature.glove as glove


class GloVeTest(unittest.TestCase):
    test_vec_file = 'tests/ml/feature/vectors.txt.gz'
    test_vocab_file = 'tests/ml/feature/vocab.txt.gz'

    def test_load_vector_array(self):
        w, w2i, i2w = glove.load_vector_array(self.test_vec_file, self.test_vocab_file)
        np.testing.assert_array_equal(w, np.array([[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0]]))
        self.assertDictEqual(w2i, {'a': 1, 'the': 0})
        self.assertDictEqual(i2w, {1: 'a', 0: 'the'})

    def test_load_pre_trained_vector_array(self):
        w, w2i, i2w = glove.load_pre_trained_vector_array(self.test_vec_file)
        np.testing.assert_array_equal(w, np.array([[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]]))
        self.assertDictEqual(w2i, {'a': 0, 'the': 1})
        self.assertDictEqual(i2w, {0: 'a', 1: 'the'})
