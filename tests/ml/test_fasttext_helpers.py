import numpy as np
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
import unittest

import cocoscore.ml.fasttext_helpers as fth


class CVTest(unittest.TestCase):
    train_path = 'tests/ml/ft_simple_test.txt'
    ft_path = 'fasttext'
    model_path = 'fastText_test_model'
    test_path = train_path
    probability_path = 'ft_simple_prob.txt.gz'
    pretrained_vectors_path = 'tests/ml/pretrained.vec'

    cv_test_path = 'tests/ml/ft_simple_cv.txt'

    def test_train_call_parameters(self):
        train_call, compress_call = fth.get_fasttext_train_calls(self.train_path, {'-aaa': 1.0}, self.ft_path,
                                                                 self.model_path, thread=5,
                                                                 pretrained_vectors_path=self.pretrained_vectors_path)
        expected_train_call = self.ft_path + ' supervised -input ' + self.train_path + ' -output ' + self.model_path + \
            ' -aaa 1.0 -thread 5 -pretrainedVectors ' + self.pretrained_vectors_path
        self.assertEqual(' '.join(train_call), expected_train_call)
        expected_compress_call = self.ft_path + ' quantize -input ' + self.model_path + ' -output ' + self.model_path
        self.assertEqual(' '.join(compress_call), expected_compress_call)

    def test_test_call_parameters(self):
        predict_call = fth.get_fasttext_test_calls(self.test_path, self.ft_path, self.model_path)
        expected_predict_call = self.ft_path + ' predict-prob ' + self.model_path + ' ' + self.test_path + ' ' + \
            str(2)
        self.assertEqual(' '.join(predict_call), expected_predict_call)

    def test_fit(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        expected_model_path = self.model_path + '.bin'
        self.assertEqual(model_path, expected_model_path)
        self.assertTrue(os.path.isfile(model_path))
        os.remove(model_path)

    def test_fit_compressed(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000, '-wordNgrams': 2}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=True)
        expected_model_path = self.model_path + '.ftz'
        self.assertEqual(model_path, expected_model_path)
        self.assertTrue(os.path.isfile(model_path))
        os.remove(model_path)

    def test_fit_pretrained_vectors(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False, pretrained_vectors_path=self.pretrained_vectors_path)
        expected_model_path = self.model_path + '.bin'
        self.assertEqual(model_path, expected_model_path)
        self.assertTrue(os.path.isfile(model_path))
        os.remove(model_path)

    def test_predict(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        fth.fasttext_predict(model_path, self.test_path, self.ft_path, self.probability_path)
        self.assertTrue(os.path.isfile(self.probability_path))
        os.remove(model_path)
        os.remove(self.probability_path)

    def test_fasttext_class_probabilities(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        fth.fasttext_predict(model_path, self.test_path, self.ft_path, self.probability_path)
        probabilities = fth.load_fasttext_class_probabilities(self.probability_path)
        self.assertEqual(len(probabilities), 40)
        self.assertTrue(all([x > 0.75 for x in probabilities[:20]]))
        self.assertTrue(all([x < 0.25 for x in probabilities[20:]]))
        os.remove(model_path)
        os.remove(self.probability_path)

    def test_fasttext_load_labels(self):
        labels = fth.load_labels(self.train_path, compression=False)
        self.assertEqual(len(labels), 40)
        self.assertTrue(all([x == 1 for x in labels[:20]]))
        self.assertTrue(all([x == 0 for x in labels[20:]]))

    def test_fasttext_cv_independent_associations(self):
        dim = 20
        bucket = 1000
        cv_folds = 2
        cv_results = fth.fasttext_cv_independent_associations(self.cv_test_path, {'-bucket': bucket, '-dim': dim},
                                                              self.ft_path,
                                                              cv_folds=cv_folds, random_state=np.random.RandomState(0))
        expected_col_names = [
            'dim',
            'bucket',
            'mean_test_score',
            'std_test_score',
            'mean_train_score',
            'std_train_score',
            'split0_test_score',
            'split0_train_score',
            'split0_n_test',
            'split0_pos_test',
            'split0_n_train',
            'split0_pos_train',
            'split1_test_score',
            'split1_train_score',
            'split1_n_test',
            'split1_pos_test',
            'split1_n_train',
            'split1_pos_train',
        ]
        expected_values = [
            [dim] * cv_folds,
            [bucket] * cv_folds,
            [1.0] * cv_folds,
            [0.0] * cv_folds,
            [1.0] * cv_folds,
            [0.0] * cv_folds,
            [1.0] * cv_folds,
            [1.0] * cv_folds,
            [20] * cv_folds,
            [0.5] * cv_folds,
            [20] * cv_folds,
            [0.5] * cv_folds,
            [1.0] * cv_folds,
            [1.0] * cv_folds,
            [20] * cv_folds,
            [0.5] * cv_folds,
            [20] * cv_folds,
            [0.5] * cv_folds,
        ]
        expected_df = pd.DataFrame({col: values for col, values in zip(expected_col_names, expected_values)},
                                   columns=expected_col_names)
        assert_frame_equal(cv_results, expected_df)



if __name__ == '__main__':
    unittest.main()
