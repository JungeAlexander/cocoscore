import pandas as pd
from pytest import raises

from cocoscore.ml.distance_scores import _distance_scorer
from cocoscore.ml.distance_scores import reciprocal_distance
from cocoscore.tools.data_tools import load_data_frame


class TestClass(object):
    paragraph_scores_file = 'tests/ml/paragraph_distance_scores_file.tsv'
    document_scores_file = 'tests/ml/document_distance_scores_file.tsv'

    def test_paragraph_scores_reciprocal(self):
        paragraph_df = load_data_frame(self.paragraph_scores_file, class_labels=False, match_distance=True)
        scores = reciprocal_distance(paragraph_df)
        expected = pd.Series([1., .1, .01, .001])
        pd.testing.assert_series_equal(scores, expected, check_names=False)

    def test_document_scores_reciprocal(self):
        document_df = load_data_frame(self.document_scores_file, class_labels=False, match_distance=True)
        scores = reciprocal_distance(document_df)
        expected = pd.Series([1., .5, 1 / 3])
        pd.testing.assert_series_equal(scores, expected, check_names=False)

    def test_distance_scorer_exception(self):
        with raises(ValueError):
            _distance_scorer(load_data_frame(self.paragraph_scores_file, class_labels=False, match_distance=False),
                             None)
