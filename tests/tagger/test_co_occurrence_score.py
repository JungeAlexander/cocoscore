import unittest

import numpy
import pandas
from pandas.util.testing import assert_frame_equal

from cocoscore.ml.distance_scores import polynomial_decay_distance
from cocoscore.ml.fasttext_helpers import fasttext_fit_predict_default
import cocoscore.tagger.co_occurrence_score as co_occurrence_score
import cocoscore.tools.data_tools as dt


def assert_deep_almost_equal(test_case, expected, actual, *args, **kwargs):
    """
    See https://stackoverflow.com/questions/23549419/assert-that-two-dictionaries-are-almost-equal#23550280
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    :param test_case: TestCase object on which we can call all of the basic
    'assert' methods.
    :param expected result
    :param actual result
    :type test_case: :py:class:`unittest.TestCase` object
    """
    # is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, complex)):
            test_case.assertAlmostEqual(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, numpy.ndarray)):
            test_case.assertEqual(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assert_deep_almost_equal(test_case, v1, v2,
                                         __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            test_case.assertEqual(set(expected), set(actual))
            for key in expected:
                assert_deep_almost_equal(test_case, expected[key], actual[key],
                                         __trace=repr(key), *args, **kwargs)
        else:
            test_case.assertEqual(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        # if is_root:
        #     trace = ' -> '.join(reversed(exc.traces))
        #     exc = AssertionError("%s\nTRACE: %s" % (exc.message, trace))
        raise exc


class CooccurrenceTest(unittest.TestCase):
    matches_file_path = 'tests/tagger/matches_file.tsv'
    matches_file_same_type_path = 'tests/tagger/matches_file_same_type.tsv'
    matches_document_level_comentions_file_path = 'tests/tagger/matches_file_document_level_comentions.tsv'
    matches_file_single_matches_path = 'tests/tagger/matches_file_single_matches.tsv'
    matches_file_cross_path = 'tests/tagger/matches_file_cross.tsv'
    matches_file_cross_fantasy_types_path = 'tests/tagger/matches_file_cross_fantasy_types.tsv'
    sentence_score_file_path = 'tests/tagger/sentence_scores_file.tsv'
    paragraph_score_file_path = 'tests/tagger/paragraph_scores_file.tsv'
    document_score_file_path = 'tests/tagger/document_scores_file.tsv'
    paragraph_sentence_score_file_path = 'tests/tagger/paragraph_sentence_scores_file.tsv'
    document_paragraph_sentence_score_file_path = 'tests/tagger/document_paragraph_sentence_scores_file.tsv'
    document_paragraph_score_file_path = 'tests/tagger/document_paragraph_scores_file.tsv'
    precedence_document_paragraph_sentence_score_file_path = \
        'tests/tagger/precedence_document_paragraph_sentence_scores_file.tsv'
    entity_file_path = 'tests/tagger/entities2.tsv.gz'
    entity_fantasy_types_file_path = 'tests/tagger/entities2_fantasy_types.tsv.gz'
    entity_file_same_type_path = 'tests/tagger/entities2_same_type.tsv.gz'

    cos_cv_test_path = 'tests/ml/cos_simple_cv.txt'

    def test_load_sentence_scores(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        self.assertDictEqual({('--D', 'A'): {(1111, 1, 2): 0.9, (1111, 2, 3): 0.5,
                                             (3333, 2, 2): 0.4, (3333, 2, 3): 0.44},
                              ('B', 'C'): {(2222, 1, 1): 0}}, sentence_scores)

    def test_load_paragraph_scores(self):
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path)
        self.assertDictEqual({('--D', 'A'): {(1111, 1): 0.9, (1111, 2): 0.5,
                                             (3333, 2): 0.4},
                              ('B', 'C'): {(2222, 1): 0}}, paragraph_scores)

    def test_load_document_scores(self):
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path)
        self.assertDictEqual({('--D', 'A'): {1111: 1,
                                             3333: 2},
                              ('B', 'C'): {2222: 3}}, document_scores)

    def test_weighted_counts_sentences(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, None, None, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=0,
                                                                  sentence_weight=1.0)
        self.assertDictEqual({('--D', 'A'): 15.9 + 15.44,
                              ('B', 'C'): 15,
                              'A': 15.9 + 15.44,
                              '--D': 15.9 + 15.44,
                              'B': 15,
                              'C': 15,
                              None: 15.9 + 15.44 + 15}, weighted_counts)

    def test_weighted_counts_sentences_paragraphs(self):
        scores = co_occurrence_score.load_score_file(self.paragraph_sentence_score_file_path)
        sentence_scores, paragraph_scores, _ = co_occurrence_score.split_scores(scores)
        weighted_counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, paragraph_scores, None, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=1.0,
                                                                  sentence_weight=1.0)
        self.assertDictEqual({('--D', 'A'): 15.9 + 0.9 + 15.44 + 0.4,
                              ('B', 'C'): 15,
                              'A': 15.9 + 0.9 + 15.44 + 0.4,
                              '--D': 15.9 + 0.9 + 15.44 + 0.4,
                              'B': 15,
                              'C': 15,
                              None: 15.9 + 0.9 + 15.44 + 0.4 + 15}, weighted_counts)

    def test_weighted_counts_paragraphs(self):
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(None, None, paragraph_scores, None, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=1.0,
                                                                  sentence_weight=1.0)
        assert_deep_almost_equal(self, {('--D', 'A'): 15.0 + 0.9 + 15.0 + 0.4,
                                        ('B', 'C'): 15.0,
                                        'A': 15.0 + 0.9 + 15.0 + 0.4,
                                        '--D': 15.0 + 0.9 + 15.0 + 0.4,
                                        'B': 15.0,
                                        'C': 15.0,
                                        None: 15.0 + 0.9 + 15.0 + 0.4 + 15.0}, weighted_counts)

    def test_weighted_counts_sentences_paragraphs_documents(self):
        scores = co_occurrence_score.load_score_file(self.document_paragraph_sentence_score_file_path)
        sentence_scores, paragraph_scores, document_scores = co_occurrence_score.split_scores(scores)

        weighted_counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, paragraph_scores,
                                                                  document_scores, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=2.0, paragraph_weight=1.0,
                                                                  sentence_weight=1.0)
        self.assertDictEqual({('--D', 'A'): 0.9 + 0.9 + 1 * 2 + 0.44 + 0.4 + 2 * 2,
                              ('B', 'C'): 3 * 2,
                              'A': 0.9 + 0.9 + 1 * 2 + 0.44 + 0.4 + 2 * 2,
                              '--D': 0.9 + 0.9 + 1 * 2 + 0.44 + 0.4 + 2 * 2,
                              'B': 3 * 2,
                              'C': 3 * 2,
                              None: 0.9 + 0.9 + 1 * 2 + 0.44 + 0.4 + 2 * 2 + 3 * 2}, weighted_counts)

    def test_weighted_counts_documents(self):
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(None, None, None,
                                                                  document_scores, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=2.0, paragraph_weight=1.0,
                                                                  sentence_weight=2.0)
        self.assertDictEqual({('--D', 'A'): 1 * 2 + 2 * 2,
                              ('B', 'C'): 3 * 2,
                              'A': 1 * 2 + 2 * 2,
                              '--D': 1 * 2 + 2 * 2,
                              'B': 3 * 2,
                              'C': 3 * 2,
                              None: 1 * 2 + 2 * 2 + 3 * 2}, weighted_counts)

    def test_weighted_counts_paragraphs_documents(self):
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path,)
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(None, None, paragraph_scores,
                                                                  document_scores, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=2.0, paragraph_weight=1.0,
                                                                  sentence_weight=1.0)
        assert_deep_almost_equal(self, {('--D', 'A'): 0.9 + 1 * 2. + 0.4 + 2 * 2.,
                                        ('B', 'C'): 3 * 2.,
                                        'A': 0.9 + 1 * 2. + 0.4 + 2 * 2.,
                                        '--D': 0.9 + 1 * 2. + 0.4 + 2 * 2.,
                                        'B': 3 * 2.,
                                        'C': 3 * 2.,
                                        None: 0.9 + 1 * 2. + 0.4 + 2 * 2. + 3 * 2.}, weighted_counts)

    def test_co_occurrence_score_sentences(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, None, None, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(None, self.sentence_score_file_path, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_co_occurrence_score_sentences_paragraphs(self):
        scores = co_occurrence_score.load_score_file(self.paragraph_sentence_score_file_path)
        sentence_scores, paragraph_scores, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 1.0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, paragraph_scores, None, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(None, self.paragraph_sentence_score_file_path, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_co_occurrence_score_sentences_documents(self):
        scores = co_occurrence_score.load_score_file(self.document_paragraph_sentence_score_file_path)
        sentence_scores, paragraph_scores, document_scores = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 1.0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, paragraph_scores, document_scores, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(None, self.document_paragraph_sentence_score_file_path, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_co_occurrence_score_precedence_sentences_paragraphs_documents(self):
        scores = co_occurrence_score.load_score_file(self.precedence_document_paragraph_sentence_score_file_path)
        sentence_scores, paragraph_scores, document_scores = co_occurrence_score.split_scores(scores)
        document_weight = 2.0
        paragraph_weight = 1.0
        sentence_weight = 1.0
        weighted_counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, paragraph_scores,
                                                                  document_scores, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=document_weight,
                                                                  paragraph_weight=paragraph_weight,
                                                                  sentence_weight=sentence_weight,
                                                                  ignore_scores=True)
        weight_sum = document_weight + paragraph_weight + sentence_weight
        self.assertDictEqual({('B', 'C'): weight_sum,
                              'B': weight_sum,
                              'C': weight_sum,
                              None: weight_sum}, weighted_counts)

    def test_weighted_counts_sentences_only_diseases(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, None, None, None,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=0,
                                                                  sentence_weight=1.0,
                                                                  ignore_scores=True)
        self.assertDictEqual({('--D', 'A'): 32,
                              ('B', 'C'): 16,
                              'A': 32,
                              '--D': 32,
                              'B': 16,
                              'C': 16,
                              None: 48}, weighted_counts)

    def test_co_occurrence_score_sentences_only_diseases(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, None, None, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0,
                                                         ignore_scores=True)
        scores = co_occurrence_score.co_occurrence_score(None, self.sentence_score_file_path, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent,
                                                         ignore_scores=True)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_weighted_counts_matches_file(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(self.matches_file_path, sentence_scores, None, None,
                                                                  self.entity_file_path,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=0,
                                                                  sentence_weight=1.0)
        self.assertAlmostEqual(15.9 + 15.44 + 15., weighted_counts[None])  # needed due to floating point strangeness
        del weighted_counts[None]
        self.assertDictEqual({('--D', 'A'): 15.9 + 15.44,
                              ('B', 'C'): 15.,
                              'A': 15.9 + 15.44,
                              '--D': 15.9 + 15.44,
                              'B': 15.,
                              'C': 15.}, weighted_counts)

    def test_co_occurrence_score_matches_file(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_path, sentence_scores, None, None,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(self.matches_file_path, self.sentence_score_file_path,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_co_occurrence_score_matches_file_same_type(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_same_type_path, sentence_scores, None, None,
                                                         self.entity_file_same_type_path,
                                                         first_type=2, second_type=2,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(self.matches_file_same_type_path,
                                                         self.sentence_score_file_path,
                                                         self.entity_file_same_type_path,
                                                         first_type=2, second_type=2,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_co_occurrence_score_matches_file_diseases(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        document_weight = 15.0
        paragraph_weight = 0
        sentence_weight = 1.0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_path, sentence_scores, None, None,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0,
                                                         ignore_scores=True)

        scores = co_occurrence_score.co_occurrence_score_diseases(self.matches_file_path,
                                                                  self.entity_file_path,
                                                                  document_weight=document_weight,
                                                                  sentence_weight=sentence_weight)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_weighted_counts_matches_document_level_comentions_file(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(self.matches_document_level_comentions_file_path,
                                                                  sentence_scores, None, None,
                                                                  self.entity_file_path,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=0,
                                                                  sentence_weight=1.0)

        self.assertDictEqual({('--D', 'A'): 15. + 15.44,
                              ('B', 'C'): 15.,
                              'A': 15. + 15.44,
                              '--D': 15. + 15.44,
                              'B': 15.,
                              'C': 15.,
                              None: 15. + 15.44 + 15.}, weighted_counts)

    def test_co_occurrence_score_matches_document_level_comentions_file(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_document_level_comentions_file_path,
                                                         sentence_scores, None, None,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(self.matches_document_level_comentions_file_path,
                                                         self.sentence_score_file_path,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_co_occurrence_score_matches_document_level_comentions_file_diseases(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        sentence_weight = 1.0
        counts = co_occurrence_score.get_weighted_counts(self.matches_document_level_comentions_file_path,
                                                         sentence_scores, None, None, self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=sentence_weight,
                                                         ignore_scores=True)

        scores = co_occurrence_score.co_occurrence_score_diseases(self.matches_document_level_comentions_file_path,
                                                                  self.entity_file_path,
                                                                  document_weight=document_weight,
                                                                  sentence_weight=sentence_weight)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_weighted_counts_matches_single_matches_file(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(self.matches_file_single_matches_path,
                                                                  sentence_scores, None, None,
                                                                  self.entity_file_path,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=0,
                                                                  sentence_weight=1.0)
        self.assertAlmostEqual(15.9 + 15.44 + 15., weighted_counts[None])  # needed due to floating point strangeness
        del weighted_counts[None]
        self.assertDictEqual({('--D', 'A'): 15.9 + 15.44,
                              ('B', 'C'): 15.,
                              'A': 15.9 + 15.44,
                              '--D': 15.9 + 15.44,
                              'B': 15.,
                              'C': 15.}, weighted_counts)

    def test_co_occurrence_score_matches_single_matches_file(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_single_matches_path,
                                                         sentence_scores, None, None,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(self.matches_file_single_matches_path,
                                                         self.sentence_score_file_path,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_co_occurrence_score_matches_single_matches_file_diseases(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        sentence_weight = 1.0

        counts = co_occurrence_score.get_weighted_counts(self.matches_file_single_matches_path,
                                                         sentence_scores, None, None, self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=sentence_weight,
                                                         ignore_scores=True)

        scores = co_occurrence_score.co_occurrence_score_diseases(self.matches_file_path,
                                                                  self.entity_file_path,
                                                                  document_weight=document_weight,
                                                                  sentence_weight=sentence_weight)
        c_a_d = counts[('--D', 'A')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])

    def test_weighted_counts_matches_file_cross(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        weighted_counts = co_occurrence_score.get_weighted_counts(self.matches_file_cross_path, sentence_scores,
                                                                  None, None,
                                                                  self.entity_file_path,
                                                                  first_type=9606, second_type=-26,
                                                                  document_weight=15.0, paragraph_weight=0,
                                                                  sentence_weight=1.0)
        self.assertAlmostEqual(15.9 + 15.44 + 15. + 15., weighted_counts[None])  # needed due to float inaccuracy
        del weighted_counts[None]
        self.assertAlmostEqual(15.9 + 15.44 + 15., weighted_counts['--D'])
        del weighted_counts['--D']
        self.assertDictEqual({('--D', 'A'): 15.9 + 15.44,
                              ('--D', 'B'): 15.,
                              ('B', 'C'): 15.,
                              'A': 15.9 + 15.44,
                              'B': 15. + 15.,
                              'C': 15.}, weighted_counts)

    def test_co_occurrence_score_matches_file_cross(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_cross_path, sentence_scores, None, None,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(self.matches_file_cross_path, self.sentence_score_file_path,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_d_b = counts[('--D', 'B')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        s_d_b = c_d_b ** weighting_exponent * ((c_d_b * c_all) / (c_b * c_d)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])
        self.assertAlmostEqual(s_d_b, scores[('--D', 'B')])

    def test_co_occurrence_score_matches_file_cross_swap_types(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_cross_path, sentence_scores,
                                                         None, None,
                                                         self.entity_file_path,
                                                         first_type=-26, second_type=9606,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(self.matches_file_cross_path, self.sentence_score_file_path,
                                                         self.entity_file_path,
                                                         first_type=-26, second_type=9606,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_d_b = counts[('--D', 'B')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        s_d_b = c_d_b ** weighting_exponent * ((c_d_b * c_all) / (c_b * c_d)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])
        self.assertAlmostEqual(s_d_b, scores[('--D', 'B')])

    def test_co_occurrence_score_matches_file_cross_fantasy_types(self):
        scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        sentence_scores, _, _ = co_occurrence_score.split_scores(scores)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_cross_fantasy_types_path, sentence_scores,
                                                         None, None,
                                                         self.entity_fantasy_types_file_path,
                                                         first_type=1, second_type=2,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(self.matches_file_cross_fantasy_types_path,
                                                         self.sentence_score_file_path,
                                                         self.entity_fantasy_types_file_path,
                                                         first_type=1, second_type=2,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         weighting_exponent=weighting_exponent)
        c_a_d = counts[('--D', 'A')]
        c_d_b = counts[('--D', 'B')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        s_d_b = c_d_b ** weighting_exponent * ((c_d_b * c_all) / (c_b * c_d)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])
        self.assertAlmostEqual(s_d_b, scores[('--D', 'B')])

    def test_co_occurrence_score_matches_file_cross_diseases(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        sentence_weight = 1.0
        counts = co_occurrence_score.get_weighted_counts(self.matches_file_cross_path, sentence_scores,
                                                         None, None,
                                                         self.entity_file_path,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=sentence_weight,
                                                         ignore_scores=True)

        scores = co_occurrence_score.co_occurrence_score_diseases(self.matches_file_cross_path,
                                                                  self.entity_file_path,
                                                                  document_weight=document_weight,
                                                                  sentence_weight=sentence_weight)
        c_a_d = counts[('--D', 'A')]
        c_d_b = counts[('--D', 'B')]
        c_a = counts['A']
        c_d = counts['--D']
        c_all = counts[None]
        s_a_d = c_a_d ** weighting_exponent * ((c_a_d * c_all) / (c_a * c_d)) ** (1 - weighting_exponent)
        c_b_c = counts[('B', 'C')]
        c_b = counts['B']
        c_c = counts['C']
        s_b_c = c_b_c ** weighting_exponent * ((c_b_c * c_all) / (c_b * c_c)) ** (1 - weighting_exponent)
        s_d_b = c_d_b ** weighting_exponent * ((c_d_b * c_all) / (c_b * c_d)) ** (1 - weighting_exponent)
        self.assertAlmostEqual(s_a_d, scores[('--D', 'A')])
        self.assertAlmostEqual(s_b_c, scores[('B', 'C')])
        self.assertAlmostEqual(s_d_b, scores[('--D', 'B')])

    def test_cocoscore_cv_independent_associations(self):
        sentence_weight = 1
        paragraph_weight = 1
        document_weight = 1
        cv_folds = 2
        test_df = dt.load_data_frame(self.cos_cv_test_path, match_distance=True)
        test_df['text'] = test_df['text'].apply(lambda s: s.strip().lower())
        cv_results = co_occurrence_score.cv_independent_associations(test_df,
                                                                     {'sentence_weight': sentence_weight,
                                                                      'paragraph_weight': paragraph_weight,
                                                                      'document_weight': document_weight,
                                                                      },
                                                                     cv_folds=cv_folds,
                                                                     random_state=numpy.random.RandomState(3),
                                                                     fasttext_epochs=5,
                                                                     fasttext_bucket=1000,
                                                                     fasttext_dim=20)
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
            [24] * cv_runs,
            [0.5] * cv_runs,
            [24] * cv_runs,
            [0.5] * cv_runs,
            [1.0] * cv_runs,
            [1.0] * cv_runs,
            [24] * cv_runs,
            [0.5] * cv_runs,
            [24] * cv_runs,
            [0.5] * cv_runs,
        ]
        expected_df = pandas.DataFrame({col: values for col, values in zip(expected_col_names, expected_values)},
                                       columns=expected_col_names)
        assert_frame_equal(cv_results, expected_df)

    def test_cocoscore_cv_independent_associations_bad_param(self):
        test_df = dt.load_data_frame(self.cos_cv_test_path, match_distance=True)
        test_df['text'] = test_df['text'].apply(lambda s: s.strip().lower())
        with self.assertRaises(TypeError) as cm:
            _ = co_occurrence_score.cv_independent_associations(test_df, {'sentence_weightXXXX': 1,
                                                                          'paragraph_weight': 1,
                                                                          'document_weight': 1,
                                                                          },
                                                                cv_folds=2,
                                                                random_state=numpy.random.RandomState(3),
                                                                fasttext_epochs=5,
                                                                fasttext_bucket=1000,
                                                                fasttext_dim=20,
                                                                constant_scoring='document')
        self.assertEqual(cm.exception.args[0],
                         "co_occurrence_score() got an unexpected keyword argument 'sentence_weightXXXX'")


    def test_cocoscore_cv_independent_associations_bad_constant_scoring(self):
        test_df = dt.load_data_frame(self.cos_cv_test_path, match_distance=True)
        test_df['text'] = test_df['text'].apply(lambda s: s.strip().lower())
        with self.assertRaises(ValueError) as cm:
            _ = co_occurrence_score.cv_independent_associations(test_df, {'sentence_weight': 1,
                                                                          'paragraph_weight': 1,
                                                                          'document_weight': 1,
                                                                          },
                                                                cv_folds=2,
                                                                random_state=numpy.random.RandomState(3),
                                                                fasttext_epochs=5,
                                                                fasttext_bucket=1000,
                                                                fasttext_dim=20,
                                                                constant_scoring='documenti')
        self.assertEqual(cm.exception.args[0], 'Unknown constant_scoring parameter: documenti')

    def test_cocoscore_constant_sentence_scoring(self):
        df = dt.load_data_frame(self.cos_cv_test_path, match_distance=True)
        df['text'] = df['text'].apply(lambda s: s.strip().lower())
        train_df = df.copy()
        test_df = df.copy()
        fasttext_function = lambda train, valid, epochs, dim, bucket: fasttext_fit_predict_default(train, valid,
                                                                                                   epochs=epochs,
                                                                                                   dim=dim,
                                                                                                   bucket=bucket)
        def new_match_distance_function(data_frame):
            return polynomial_decay_distance(data_frame, 0, -2)
        train_scores, test_scores = co_occurrence_score._get_train_test_scores(train_df, test_df, fasttext_function,
                                                                               fasttext_epochs=5, fasttext_dim=20,
                                                                               fasttext_bucket=1000,
                                                                               match_distance_function=
                                                                               new_match_distance_function,
                                                                               constant_scoring='sentence')
        sentence_matches = numpy.logical_and(df['sentence'] != -1, df['paragraph'] != -1)
        non_sentence_matches = numpy.logical_not(sentence_matches)
        for scores in (train_scores, test_scores):
            self.assertTrue((scores[sentence_matches] == 1).all())
            self.assertTrue((scores[non_sentence_matches] == -1).all())

    def test_cocoscore_constant_paragraph_scoring(self):
        df = dt.load_data_frame(self.cos_cv_test_path, match_distance=True)
        df['text'] = df['text'].apply(lambda s: s.strip().lower())
        train_df = df.copy()
        test_df = df.copy()
        fasttext_function = lambda train, valid, epochs, dim, bucket: fasttext_fit_predict_default(train, valid,
                                                                                                   epochs=epochs,
                                                                                                   dim=dim,
                                                                                                   bucket=bucket)
        def new_match_distance_function(data_frame):
            return polynomial_decay_distance(data_frame, 0, -2)
        train_scores, test_scores = co_occurrence_score._get_train_test_scores(train_df, test_df, fasttext_function,
                                                                               fasttext_epochs=5, fasttext_dim=20,
                                                                               fasttext_bucket=1000,
                                                                               match_distance_function=
                                                                               new_match_distance_function,
                                                                               constant_scoring='paragraph')

        paragraph_matches = numpy.logical_and(df['sentence'] == -1, df['paragraph'] != -1)
        document_matches = numpy.logical_and(df['sentence'] == -1, df['paragraph'] == -1)
        for scores in (train_scores, test_scores):
            self.assertTrue((scores[paragraph_matches] == 1).all())
            self.assertTrue((scores[document_matches] == -1).all())

    def test_cocoscore_constant_document_scoring(self):
        df = dt.load_data_frame(self.cos_cv_test_path, match_distance=True)
        df['text'] = df['text'].apply(lambda s: s.strip().lower())
        train_df = df.copy()
        test_df = df.copy()
        fasttext_function = lambda train, valid, epochs, dim, bucket: fasttext_fit_predict_default(train, valid,
                                                                                                   epochs=epochs,
                                                                                                   dim=dim,
                                                                                                   bucket=bucket)
        def new_match_distance_function(data_frame):
            return polynomial_decay_distance(data_frame, 0, -2)
        train_scores, test_scores = co_occurrence_score._get_train_test_scores(train_df, test_df, fasttext_function,
                                                                               fasttext_epochs=5, fasttext_dim=20,
                                                                               fasttext_bucket=1000,
                                                                               match_distance_function=
                                                                               new_match_distance_function,
                                                                               constant_scoring='document')

        paragraph_matches = numpy.logical_and(df['sentence'] == -1, df['paragraph'] != -1)
        document_matches = numpy.logical_and(df['sentence'] == -1, df['paragraph'] == -1)
        for scores in (train_scores, test_scores):
            self.assertTrue((scores[paragraph_matches] == -1).all())
            self.assertTrue((scores[document_matches] == 1).all())
