import os

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

import cocoscore.ml.fasttext_helpers as fth
import cocoscore.tools.data_tools as dt


class TestClass(object):
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
        assert ' '.join(train_call) == expected_train_call
        expected_compress_call = self.ft_path + ' quantize -input ' + self.model_path + ' -output ' + self.model_path
        assert ' '.join(compress_call) == expected_compress_call

    def test_test_call_parameters(self):
        predict_call = fth.get_fasttext_test_calls(self.test_path, self.ft_path, self.model_path)
        expected_predict_call = self.ft_path + ' predict-prob ' + self.model_path + ' ' + self.test_path + ' ' + \
            str(2)
        assert ' '.join(predict_call) == expected_predict_call

    def test_fit(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        expected_model_path = self.model_path + '.bin'
        assert model_path == expected_model_path
        assert os.path.isfile(model_path)
        os.remove(model_path)

    def test_fit_compressed(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000, '-wordNgrams': 2}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=True)
        expected_model_path = self.model_path + '.ftz'
        assert model_path == expected_model_path
        assert os.path.isfile(model_path)
        os.remove(model_path)

    def test_fit_pretrained_vectors(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False, pretrained_vectors_path=self.pretrained_vectors_path)
        expected_model_path = self.model_path + '.bin'
        assert model_path == expected_model_path
        assert os.path.isfile(model_path)
        os.remove(model_path)

    def test_predict(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        fth.fasttext_predict(model_path, self.test_path, self.ft_path, self.probability_path)
        assert os.path.isfile(self.probability_path)
        os.remove(model_path)
        os.remove(self.probability_path)

    def test_fasttext_class_probabilities(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        fth.fasttext_predict(model_path, self.test_path, self.ft_path, self.probability_path)
        probabilities = fth.load_fasttext_class_probabilities(self.probability_path)
        assert len(probabilities) == 40
        assert all([x > 0.75 for x in probabilities[:20]])
        assert all([x < 0.25 for x in probabilities[20:]])
        os.remove(model_path)
        os.remove(self.probability_path)

    def test_fasttext_load_labels(self):
        labels = fth.load_labels(self.train_path, compression=False)
        assert len(labels) == 40
        assert all([x == 1 for x in labels[:20]])
        assert all([x == 0 for x in labels[20:]])

    def test_fasttext_cv_independent_associations(self):
        dim = 20
        bucket = 1000
        cv_folds = 2
        test_df = dt.load_data_frame(self.cv_test_path)
        test_df['text'] = test_df['text'].apply(lambda s: s.strip().lower())
        cv_results = fth.fasttext_cv_independent_associations(test_df, {'-bucket': bucket, '-dim': dim},
                                                              self.ft_path,
                                                              cv_folds=cv_folds, random_state=np.random.RandomState(3))
        expected_col_names = [
            'mean_test_score',
            'stdev_test_score',
            'mean_train_score',
            'stdev_train_score',
            'split_0_test_score',
            'split_0_train_score',
            'split_0_n_test',
            'split_0_pos_test',
            'split_0_n_train',
            'split_0_pos_train',
            'split_1_test_score',
            'split_1_train_score',
            'split_1_n_test',
            'split_1_pos_test',
            'split_1_n_train',
            'split_1_pos_train',
        ]
        cv_runs = 1
        expected_values = [
            [1.0] * cv_runs,
            [0.0] * cv_runs,
            [1.0] * cv_runs,
            [0.0] * cv_runs,
            [1.0] * cv_runs,
            [1.0] * cv_runs,
            [20] * cv_runs,
            [0.5] * cv_runs,
            [20] * cv_runs,
            [0.5] * cv_runs,
            [1.0] * cv_runs,
            [1.0] * cv_runs,
            [20] * cv_runs,
            [0.5] * cv_runs,
            [20] * cv_runs,
            [0.5] * cv_runs,
        ]
        expected_df = pd.DataFrame({col: values for col, values in zip(expected_col_names, expected_values)},
                                   columns=expected_col_names)
        assert_frame_equal(cv_results, expected_df)
