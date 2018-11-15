import numpy as np

from cocoscore.ml.feature import embedding_matrix
from cocoscore.ml.feature import glove


class TestClass(object):
    test_vec_file = 'tests/ml/feature/vectors.txt.gz'
    test_vocab_file = 'tests/ml/feature/vocab.txt.gz'
    test_vec_file_dg = 'tests/ml/feature/vectors_disease_gene.txt.gz'
    test_vocab_file_dg = 'tests/ml/feature/vocab_disease_gene.txt.gz'

    def test_generate_embedding_matrix(self):
        w, w2i, i2w = glove.load_vector_array(self.test_vec_file, self.test_vocab_file)
        m = embedding_matrix.generate_embedding_matrix([['a', 'the'], ['the', 'a']], w, w2i, max_length=2,
                                                       min_words_mapped=0, replace_disease_gene_tokens=False)
        np.testing.assert_array_equal(m, np.array([[[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]],
                                                   [[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0]]]))

    def test_generate_embedding_matrix_length_three(self):
        w, w2i, i2w = glove.load_vector_array(self.test_vec_file, self.test_vocab_file)
        m = embedding_matrix.generate_embedding_matrix([['a', 'the'], ['the', 'a']], w, w2i, max_length=3,
                                                       min_words_mapped=0, replace_disease_gene_tokens=False)
        np.testing.assert_array_equal(m, np.array([[[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0] * 4],
                                                   [[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0], [0] * 4]]))

    def test_generate_embedding_matrix_min_words_three(self):
        w, w2i, i2w = glove.load_vector_array(self.test_vec_file, self.test_vocab_file)
        m = embedding_matrix.generate_embedding_matrix([['a', 'the'], ['the', 'a']], w, w2i, max_length=2,
                                                       min_words_mapped=3, replace_disease_gene_tokens=False)
        np.testing.assert_array_equal(m, np.array([[[np.nan] * 4, [np.nan] * 4],
                                                   [[np.nan] * 4, [np.nan] * 4]]))

    def test_generate_embedding_matrix_disease_gene(self):
        w, w2i, i2w = glove.load_vector_array(self.test_vec_file_dg, self.test_vocab_file_dg)
        m = embedding_matrix.generate_embedding_matrix([['mygenetoken', 'a'], ['the', 'mydiseasetoken']], w, w2i,
                                                       max_length=2,
                                                       min_words_mapped=1, replace_disease_gene_tokens=True)
        np.testing.assert_array_equal(m, np.array([[[0, 0, 0, 1], [1, 0, 0, 0]],
                                                   [[0.5, 0.5, 0.5, 0.5], [0, 0, 1, 0]]]))
