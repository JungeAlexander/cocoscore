import numpy as np
import pandas
from pandas.util.testing import assert_frame_equal
import unittest

import cocoscore.ml.cv as cv
import cocoscore.ml.fasttext_helpers as fth
import cocoscore.tagger.co_occurrence_score as cos
import cocoscore.tools.data_tools as data_tools


class CVTest(unittest.TestCase):
    testcase_df = pandas.read_csv('tests/ml/test_df.tsv', sep='\t', header=0, index_col=None)
    testcase_cross_df = pandas.read_csv('tests/ml/test_cross_df.tsv', sep='\t', header=0, index_col=None)
    testcase_cv_fold_stats = pandas.read_csv('tests/ml/test_cv_fold_stats.csv', sep=',', header=0, index_col=None)
    test_case_df_path = 'tests/ml/cv_test.tsv'

    ft_path = 'fasttext'
    ft_cv_test_path = 'tests/ml/ft_simple_cv.txt'
    cos_cv_test_path = 'tests/ml/cos_simple_cv.txt'

    def test_reproducibility(self):
        run1 = cv.cv_independent_entities(self.testcase_df, random_state=np.random.RandomState(0))
        run2 = cv.cv_independent_entities(self.testcase_df, random_state=np.random.RandomState(0))
        for first, second in zip(run1, run2):
            train_first, test_first = first
            train_second, test_second = second
            np.testing.assert_array_equal(train_first, train_second)
            np.testing.assert_array_equal(test_first, test_second)

    def test_randomness(self):
        # since this may fail due to randomness in some cases, allow a couple of attempts
        max_attempts = 20
        attempt = 0
        while True:
            if attempt == max_attempts:
                self.fail('Failed due since no randomness in shuffled splits found.')
            else:
                try:
                    run1 = cv.cv_independent_entities(self.testcase_df)
                    run2 = cv.cv_independent_entities(self.testcase_df)
                    for first, second in zip(run1, run2):
                        train_first, test_first = first
                        train_second, test_second = second
                        self.assertFalse(all([x in train_first for x in train_second]))
                        self.assertFalse(all([x in test_first for x in test_second]))
                    break
                except AssertionError:
                    attempt += 1

    def test_cv_fold_count(self):
        run = cv.cv_independent_entities(self.testcase_df, cv_folds=3)
        self.assertEqual(3, sum([1 for _ in run]))

    def test_train_test_independence(self):
        _ = self.testcase_df['entity1'].unique()
        _ = self.testcase_df['entity2'].unique()
        run = cv.cv_independent_entities(self.testcase_df)
        for train, test in run:
            train_df = self.testcase_df.iloc[train, :]
            train_genes = train_df['entity1'].unique()
            train_diseases = train_df['entity2'].unique()
            test_df = self.testcase_df.iloc[test, :]
            test_genes = test_df['entity1'].unique()
            test_diseases = test_df['entity2'].unique()
            self.assertEqual(0, len(set(train_genes).intersection(set(test_genes))))
            self.assertEqual(0, len(set(train_diseases).intersection(set(test_diseases))))

    def test_train_test_independence_completeness(self):
        all_genes = self.testcase_cross_df['entity1'].unique()
        all_diseases = self.testcase_cross_df['entity2'].unique()
        run = cv.cv_independent_entities(self.testcase_cross_df, cv_folds=3, treat_entity_columns_separately=True)
        for train, test in run:
            train_df = self.testcase_cross_df.iloc[train, :]
            train_genes = train_df['entity1'].unique()
            train_diseases = train_df['entity2'].unique()
            test_df = self.testcase_cross_df.iloc[test, :]
            test_genes = test_df['entity1'].unique()
            test_diseases = test_df['entity2'].unique()
            self.assertEqual(0, len(set(train_genes).intersection(set(test_genes))))
            self.assertEqual(0, len(set(train_diseases).intersection(set(test_diseases))))
            self.assertSetEqual(set(all_genes), set(train_genes).union(set(test_genes)))
            self.assertSetEqual(set(all_diseases), set(train_diseases).union(set(test_diseases)))

    def test_compute_cv_fold_stats(self):
        cv_splits = (([0, 1], [2, 3]), ([2, 3], [0, 1]))
        expected_df = pandas.DataFrame({'fold': [0, 1],
                                        'n_train': [2, 2],
                                        'pos_train': [0, 0.5],
                                        'n_test': [2, 2],
                                        'pos_test': [0.5, 0]})
        observed_df = cv.compute_cv_fold_stats(self.testcase_cv_fold_stats, cv_splits)
        assert_frame_equal(expected_df, observed_df)

    def test_reproducibility_associations(self):
        test_case_df = data_tools.load_data_frame(self.test_case_df_path)
        run1 = cv.cv_independent_associations(test_case_df, cv_folds=3, random_state=np.random.RandomState(0))
        run2 = cv.cv_independent_associations(test_case_df, cv_folds=3, random_state=np.random.RandomState(0))
        for first, second in zip(run1, run2):
            train_first, test_first = first
            train_second, test_second = second
            np.testing.assert_array_equal(train_first, train_second)
            np.testing.assert_array_equal(test_first, test_second)
            self.assertEqual(len(train_first), 4)
            self.assertEqual(len(test_first), 2)
            self.assertEqual(len(train_second), 4)
            self.assertEqual(len(test_second), 2)

    def test_randomness_associations(self):
        test_case_df = data_tools.load_data_frame(self.test_case_df_path)
        # since this may fail due to randomness in some cases, allow a couple of attempts
        max_attempts = 20
        attempt = 0
        while True:
            if attempt == max_attempts:
                self.fail('Failed due since no randomness in shuffled splits found.')
            else:
                try:
                    run1 = cv.cv_independent_associations(test_case_df, cv_folds=3)
                    run2 = cv.cv_independent_associations(test_case_df, cv_folds=3)
                    for first, second in zip(run1, run2):
                        train_first, test_first = first
                        train_second, test_second = second
                        self.assertFalse(all([x in train_first for x in train_second]))
                        self.assertFalse(all([x in test_first for x in test_second]))
                    break
                except AssertionError:
                    attempt += 1

    def test_random_fth_parameter_sampler_reproducibility(self):
        dist1 = fth.get_hyperparameter_distributions()
        sample1 = cv.get_random_parameter_sampler(dist1, 3).__next__()
        dist2 = fth.get_hyperparameter_distributions()
        sample2 = cv.get_random_parameter_sampler(dist2, 3).__next__()
        self.assertEqual(sample1, sample2)

    def test_random_fth_parameter_sampler_reproducibility_seed_argument(self):
        dist1 = fth.get_hyperparameter_distributions(1)
        sample1 = cv.get_random_parameter_sampler(dist1, 3).__next__()
        dist2 = fth.get_hyperparameter_distributions(1)
        sample2 = cv.get_random_parameter_sampler(dist2, 3).__next__()
        self.assertEqual(sample1, sample2)

    def test_random_fth_parameter_sampler_random_seed_argument(self):
        dist1 = fth.get_hyperparameter_distributions(1)
        sample1 = cv.get_random_parameter_sampler(dist1, 3).__next__()
        dist2 = fth.get_hyperparameter_distributions(12)
        sample2 = cv.get_random_parameter_sampler(dist2, 3).__next__()
        self.assertNotEqual(sample1, sample2)

    def test_random_fth_parameter_sampler_random(self):
        dist = fth.get_hyperparameter_distributions()
        generator = cv.get_random_parameter_sampler(dist, 3)
        sample1 = generator.__next__()
        sample2 = generator.__next__()
        self.assertNotEqual(sample1, sample2)

    def test_random_cos_parameter_sampler_reproducibility(self):
        dist1 = cos.get_hyperparameter_distributions()
        sample1 = cv.get_random_parameter_sampler(dist1, 3).__next__()
        dist2 = cos.get_hyperparameter_distributions()
        sample2 = cv.get_random_parameter_sampler(dist2, 3).__next__()
        self.assertEqual(sample1, sample2)

    def test_random_cos_parameter_sampler_reproducibility_seed_argument(self):
        dist1 = cos.get_hyperparameter_distributions(1)
        sample1 = cv.get_random_parameter_sampler(dist1, 3).__next__()
        dist2 = cos.get_hyperparameter_distributions(1)
        sample2 = cv.get_random_parameter_sampler(dist2, 3).__next__()
        self.assertEqual(sample1, sample2)

    def test_random_cos_parameter_sampler_random_seed_argument(self):
        dist1 = cos.get_hyperparameter_distributions(1)
        sample1 = cv.get_random_parameter_sampler(dist1, 3).__next__()
        dist2 = fth.get_hyperparameter_distributions(12)
        sample2 = cv.get_random_parameter_sampler(dist2, 3).__next__()
        self.assertNotEqual(sample1, sample2)

    def test_random_cos_parameter_sampler_random(self):
        dist = cos.get_hyperparameter_distributions()
        generator = cv.get_random_parameter_sampler(dist, 3)
        sample1 = generator.__next__()
        sample2 = generator.__next__()
        self.assertNotEqual(sample1, sample2)

    def test_fth_random_cv(self):
        bucket = 1000
        dim = 20
        cv_folds = 2
        cv_iterations = 2

        def cv_function(data_df, params, random_state):
            return fth.fasttext_cv_independent_associations(data_df, params, self.ft_path, cv_folds=cv_folds,
                                                            random_state=random_state)

        test_df = data_tools.load_data_frame(self.ft_cv_test_path)
        test_df['text'] = test_df['text'].apply(lambda s: s.strip().lower())
        cv_results = cv.random_cv(test_df, cv_function, cv_iterations, {'-bucket': bucket, '-dim': dim},
                                  fth.get_hyperparameter_distributions(), 3)
        expected_col_names = [
            'bucket',
            'dim',
            'epoch',
            'lr',
            'wordNgrams',
            'ws',
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
        self.assertListEqual(expected_col_names, list(cv_results.columns))
        # following parameters are chosen randomly and hence cannot be tested but only that they differ between the CV
        # runs
        random_col_names = [
            'epoch',
            'lr',
            'wordNgrams',
            'ws',
        ]
        for rand in random_col_names:
            self.assertNotEqual(cv_results.loc[0, rand], cv_results.loc[1, rand])

        # ignore columns that are linked to test performance since this cannot be tested for random parameter choices
        ignore_params = [
            'mean_test_score',
            'stdev_test_score',
            'mean_train_score',
            'stdev_train_score',
            'split_0_test_score',
            'split_0_train_score',
            'split_1_test_score',
            'split_1_train_score',
        ]

        expected_values = [
            [1000] * cv_iterations,
            [20] * cv_iterations,
            [.1] * cv_iterations,
            [.2] * cv_iterations,
            [.3] * cv_iterations,
            [.4] * cv_iterations,
            [1.0] * cv_iterations,
            [0.0] * cv_iterations,
            [1.0] * cv_iterations,
            [0.0] * cv_iterations,
            [1.0] * cv_iterations,
            [1.0] * cv_iterations,
            [20] * cv_iterations,
            [0.5] * cv_iterations,
            [20] * cv_iterations,
            [0.5] * cv_iterations,
            [1.0] * cv_iterations,
            [1.0] * cv_iterations,
            [20] * cv_iterations,
            [0.5] * cv_iterations,
            [20] * cv_iterations,
            [0.5] * cv_iterations,
        ]
        expected_df = pandas.DataFrame({col: values for col, values in zip(expected_col_names, expected_values)},
                                       columns=expected_col_names)
        results_test_df = cv_results.drop(random_col_names + ignore_params, axis=1)
        expected_test_df = expected_df.drop(random_col_names + ignore_params, axis=1)
        assert_frame_equal(results_test_df, expected_test_df)

    def test_cos_random_cv_bad_param(self):
        cv_folds = 2
        cv_iterations = 2

        def cv_function(data_df, params, random_state):
            return cos.cv_independent_associations(data_df, params,
                                                   cv_folds=cv_folds, random_state=random_state, fasttext_epochs=5,
                                                   fasttext_bucket=1000, fasttext_dim=20)

        test_df = data_tools.load_data_frame(self.cos_cv_test_path, match_distance=True)
        test_df['text'] = test_df['text'].apply(lambda s: s.strip().lower())
        with self.assertRaises(TypeError) as cm:
            _ = cv.random_cv(test_df, cv_function, cv_iterations, {'sentence_weightXXXX': 1},
                            cos.get_hyperparameter_distributions(), 3)

        self.assertEqual(cm.exception.args[0],
                         "co_occurrence_score() got an unexpected keyword argument 'sentence_weightXXXX'")

    def test_cos_random_cv(self):
        paragraph_weight = 3
        cv_folds = 2
        cv_iterations = 2

        def cv_function(data_df, params, random_state):
            return cos.cv_independent_associations(data_df, params,
                                                   cv_folds=cv_folds, random_state=random_state, fasttext_epochs=5,
                                                   fasttext_bucket=1000, fasttext_dim=20)

        test_df = data_tools.load_data_frame(self.cos_cv_test_path, match_distance=True)
        test_df['text'] = test_df['text'].apply(lambda s: s.strip().lower())
        cv_results = cv.random_cv(test_df, cv_function, cv_iterations, {'paragraph_weight': paragraph_weight},
                                  cos.get_hyperparameter_distributions(), 3)

        expected_col_names = [
            'decay_rate',
            'distance_offset',
            'document_weight',
            'paragraph_weight',
            'weighting_exponent',
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
        self.assertListEqual(expected_col_names, list(cv_results.columns))
        # following parameters are chosen randomly and hence cannot be tested but only that they differ between the CV
        # runs
        random_col_names = [
            'decay_rate',
            'distance_offset',
            'document_weight',
            'weighting_exponent',
        ]
        for rand in random_col_names:
            self.assertNotEqual(cv_results.loc[0, rand], cv_results.loc[1, rand])

        # ignore columns that are linked to test performance since this cannot be tested for random parameter choices
        ignore_params = [
            'mean_test_score',
            'stdev_test_score',
            'mean_train_score',
            'stdev_train_score',
            'split_0_test_score',
            'split_0_train_score',
            'split_1_test_score',
            'split_1_train_score',
        ]

        expected_values = [
            [.333] * cv_iterations,
            [.222] * cv_iterations,
            [.111] * cv_iterations,
            [paragraph_weight] * cv_iterations,
            [.111] * cv_iterations,
            [1.0] * cv_iterations,
            [0.0] * cv_iterations,
            [1.0] * cv_iterations,
            [0.0] * cv_iterations,
            [1.0] * cv_iterations,
            [1.0] * cv_iterations,
            [24] * cv_iterations,
            [0.5] * cv_iterations,
            [24] * cv_iterations,
            [0.5] * cv_iterations,
            [1.0] * cv_iterations,
            [1.0] * cv_iterations,
            [24] * cv_iterations,
            [0.5] * cv_iterations,
            [24] * cv_iterations,
            [0.5] * cv_iterations,
        ]
        expected_df = pandas.DataFrame({col: values for col, values in zip(expected_col_names, expected_values)},
                                       columns=expected_col_names)
        results_test_df = cv_results.drop(random_col_names + ignore_params, axis=1)
        expected_test_df = expected_df.drop(random_col_names + ignore_params, axis=1)
        assert_frame_equal(results_test_df, expected_test_df)


if __name__ == '__main__':
    unittest.main()
