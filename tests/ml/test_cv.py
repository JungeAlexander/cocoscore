import numpy as np
import pandas
from pandas.util.testing import assert_frame_equal
import unittest

import cocoscore.ml.cv as cv
import cocoscore.ml.fasttext_helpers as fth
import cocoscore.tools.data_tools as data_tools


class CVTest(unittest.TestCase):
    testcase_df = pandas.read_csv('tests/ml/test_df.tsv', sep='\t', header=0, index_col=None)
    testcase_cross_df = pandas.read_csv('tests/ml/test_cross_df.tsv', sep='\t', header=0, index_col=None)
    testcase_cv_fold_stats = pandas.read_csv('tests/ml/test_cv_fold_stats.csv', sep=',', header=0, index_col=None)
    test_case_df_path = 'tests/ml/cv_test.tsv'

    ft_path = 'fasttext'
    ft_cv_test_path = 'tests/ml/ft_simple_cv.txt'

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

    def test_random_parameter_sampler_reproducibility(self):
        dist1 = fth.get_hyperparameter_distributions()
        sample1 = cv.get_random_parameter_sampler(dist1, 3).__next__()
        dist2 = fth.get_hyperparameter_distributions()
        sample2 = cv.get_random_parameter_sampler(dist2, 3).__next__()
        self.assertEqual(sample1, sample2)

    def test_random_parameter_sampler_random(self):
        dist = fth.get_hyperparameter_distributions()
        generator = cv.get_random_parameter_sampler(dist, 3)
        sample1 = generator.__next__()
        sample2 = generator.__next__()
        self.assertNotEqual(sample1, sample2)

    def test_random_cv(self):
        bucket = 1000
        dim = 20
        cv_folds = 2
        cv_iterations = 2

        def cv_function(data_df, params):
            return fth.fasttext_cv_independent_associations(data_df, params, self.ft_path, cv_folds=cv_folds,
                                                            random_state=np.random.RandomState(3))

        test_df = data_tools.load_data_frame(self.ft_cv_test_path)
        test_df['sentence_text'] = test_df['sentence_text'].apply(lambda s: s.strip().lower())
        cv_results = cv.random_cv(test_df, cv_function, cv_iterations, {'-bucket': bucket, '-dim': dim})
        expected_col_names = [
            'bucket',
            'dim',
            'epoch,'
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
        expected_values = [
            [1000] * cv_iterations,
            [10] * cv_iterations,
            [1.0] * cv_iterations,
            [1.1] * cv_iterations,
            [1.2] * cv_iterations,
            [1.3] * cv_iterations,
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
        assert_frame_equal(cv_results, expected_df)


if __name__ == '__main__':
    unittest.main()
