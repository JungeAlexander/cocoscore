import unittest

import numpy

import cocoscore.tagger.co_occurrence_score as co_occurrence_score


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
    entity_file_path = 'tests/tagger/entities2.tsv.gz'
    entity_fantasy_types_file_path = 'tests/tagger/entities2_fantasy_types.tsv.gz'
    entity_file_same_type_path = 'tests/tagger/entities2_same_type.tsv.gz'

    def test_load_sentence_scores(self):
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        self.assertDictEqual({('--D', 'A'): {(1111, 1, 2): 0.9, (1111, 2, 3): 0.5,
                                             (3333, 2, 2): 0.4, (3333, 2, 3): 0.44},
                              ('B', 'C'): {(2222, 1, 1): 0}}, sentence_scores)

    def test_load_paragraph_scores(self):
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path, paragraph_level=True)
        self.assertDictEqual({('--D', 'A'): {(1111, 1): 0.9, (1111, 2): 0.5,
                                             (3333, 2): 0.4},
                              ('B', 'C'): {(2222, 1): 0}}, paragraph_scores)

    def test_load_paragraph_scores_with_sentence_scores(self):
        with self.assertRaises(ValueError):
            co_occurrence_score.load_score_file(self.sentence_score_file_path, paragraph_level=True)

    def test_load_document_scores(self):
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path, document_level=True)
        self.assertDictEqual({('--D', 'A'): {1111: 1,
                                             3333: 2},
                              ('B', 'C'): {2222: 3}}, document_scores)

    def test_load_document_scores_with_paragraph_scores(self):
        with self.assertRaises(ValueError):
            co_occurrence_score.load_score_file(self.paragraph_score_file_path, document_level=True)


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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path, paragraph_level=True)
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
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path, paragraph_level=True)
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path, paragraph_level=True)
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path, document_level=True)
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
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path, document_level=True)
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
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path, paragraph_level=True)
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path, document_level=True)
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
        scores = co_occurrence_score.co_occurrence_score(None, self.sentence_score_file_path, None, None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path, paragraph_level=True)
        document_weight = 15.0
        paragraph_weight = 1.0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, paragraph_scores, None, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(None, self.sentence_score_file_path,
                                                         self.paragraph_score_file_path, None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        paragraph_scores = co_occurrence_score.load_score_file(self.paragraph_score_file_path, paragraph_level=True)
        document_scores = co_occurrence_score.load_score_file(self.document_score_file_path, document_level=True)
        document_weight = 15.0
        paragraph_weight = 1.0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, paragraph_scores, document_scores, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0)
        scores = co_occurrence_score.co_occurrence_score(None, self.sentence_score_file_path,
                                                         self.paragraph_score_file_path,
                                                         self.document_score_file_path, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
        document_weight = 15.0
        paragraph_weight = 0
        weighting_exponent = 0.6
        counts = co_occurrence_score.get_weighted_counts(None, sentence_scores, None, None, None,
                                                         first_type=9606, second_type=-26,
                                                         document_weight=document_weight,
                                                         paragraph_weight=paragraph_weight,
                                                         sentence_weight=1.0,
                                                         ignore_scores=True)
        scores = co_occurrence_score.co_occurrence_score(None, self.sentence_score_file_path, None, None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
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
                                                         None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
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
                                                         None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
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
                                                         self.sentence_score_file_path, None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
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
                                                         self.sentence_score_file_path, None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
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
                                                         None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
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
                                                         None, None,
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
        sentence_scores = co_occurrence_score.load_score_file(self.sentence_score_file_path)
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
                                                         self.sentence_score_file_path, None, None,
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
