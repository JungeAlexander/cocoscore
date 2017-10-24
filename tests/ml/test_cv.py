import numpy as np
import pandas
from pandas.util.testing import assert_frame_equal
import unittest

import cocoscore.ml.cv as cv


class CVTest(unittest.TestCase):
    testcase_df = pandas.read_csv('tests/ml/test_df.tsv', sep='\t', header=0, index_col=None)
    testcase_cross_df = pandas.read_csv('tests/ml/test_cross_df.tsv', sep='\t', header=0, index_col=None)
    testcase_cv_fold_stats = pandas.read_csv('tests/ml/test_cv_fold_stats.csv', sep=',', header=0, index_col=None)

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
        cv_splits = (((0, 1), (2, 3)), ((2, 3), (0, 1)))
        expected_df = pandas.DataFrame({'fold': [0, 1],
                                        'n_train': [2, 2],
                                        'pos_train': [0, 0.5],
                                        'n_test': [2, 2],
                                        'pos_test': [0.5, 0]})
        observed_df = cv.compute_cv_fold_stats(self.testcase_cv_fold_stats, cv_splits)
        assert_frame_equal(expected_df, observed_df)

if __name__ == '__main__':
    unittest.main()
