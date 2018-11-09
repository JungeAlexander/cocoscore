import numpy as np

from cocoscore.ml.feature.embedding_matrix import get_sentence_vector_array


class TestClass(object):

    sentence = [['this', 'is', 'a', 'test', '.', 'unknown']]
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
    word_to_index = {'this': 1, 'is': 0, 'a': 2, 'test': 3, '.': 4}

    def test_sentence_vector_generation(self):
        sentence_vector = get_sentence_vector_array(self.sentence, self.vectors, self.word_to_index, 1, True)
        np.testing.assert_array_equal(sentence_vector[0], np.array([0.25] * 4))

    def test_sentence_vector_generation_too_few_words(self):
        sentence_vector = get_sentence_vector_array(self.sentence, self.vectors, self.word_to_index, 5, True)
        np.testing.assert_array_equal(sentence_vector[0], np.array([np.nan] * 4))

    def test_sentence_vector_generation_keep_punctuation(self):
        sentence_vector = get_sentence_vector_array(self.sentence, self.vectors, self.word_to_index, 1, False)
        np.testing.assert_array_equal(sentence_vector[0], np.array([0.2, 0.2, 0.2, 0.4]))
