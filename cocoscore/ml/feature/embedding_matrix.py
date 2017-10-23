import numpy as np

from dicomclass.dataset.generate_dataset import disease_placeholder, gene_placeholder


def generate_embedding_matrix(sentences_tokenized, word_vectors, word_to_index, max_length, min_words_mapped=0,
                              replace_disease_gene_tokens=True):
    """
    Generate word vector matrix for a set of tokenized sentences by creating a feature matrix of word vectors.

    :param sentences_tokenized: list of list of strings - the tokenized sentences
    :param word_vectors:  dict mapping integer word indices to their word vectors
    :param word_to_index: dict mapping words to their integer index in word_vectors
    :param max_length: the maximal number of words to be be converted to word vectors in each sentence
    :param min_words_mapped: the minimal number of words (not counting tokens representing tagged diseases and genes)
    that need to be mapped to word vectors in each sentence. If fewer words are mapped in a given sentence,
    the matrix corresponding to the sentence is filled with numpy NaNs.
    :param replace_disease_gene_tokens: boolean indicating if tokens representing tagged diseases and genes
    are mapped to the word vectors for 'disease' and 'gene', respectively. If False, the tokens are ignored.
    :return: a three dimensional numpy array, first dimension is sentence, second is word, third is word vector
    """
    n_samples = len(sentences_tokenized)
    vector_dim = len(word_vectors[0])
    embedding_matrix = np.zeros((n_samples, max_length, vector_dim))
    for i, sentence in enumerate(sentences_tokenized):
        words_mapped = 0
        for j, word in enumerate(sentence):
            if j >= max_length:
                break
            if word not in word_to_index and replace_disease_gene_tokens and word == disease_placeholder.lower():
                embedding_matrix[i, j] = word_vectors[word_to_index['disease']]
            elif word not in word_to_index and replace_disease_gene_tokens and word == gene_placeholder.lower():
                embedding_matrix[i, j] = word_vectors[word_to_index['gene']]
            elif word in word_to_index:
                words_mapped += 1
                embedding_matrix[i, j] = word_vectors[word_to_index[word]]
        if words_mapped < min_words_mapped:
            embedding_matrix[i, :, :] = np.full((max_length, vector_dim), np.nan)
    return embedding_matrix
