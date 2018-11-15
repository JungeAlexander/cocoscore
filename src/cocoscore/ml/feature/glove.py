import gzip

import numpy as np

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def _load_vocabulary_dicts(vocabulary_file):
    vocabulary = []
    with gzip.open(vocabulary_file, 'rt', encoding='utf8', errors='raise') as file_in:
        for line in file_in:
            word = line.rstrip().split(' ')[0]
            vocabulary.append(word)
    word_to_index = {w: i for i, w in enumerate(vocabulary)}
    index_to_word = {i: w for w, i in word_to_index.items()}
    return word_to_index, index_to_word


def _get_vectors(vectors, word_to_index, index_to_word, normalize):
    vocabulary_size = len(word_to_index)
    vector_dim = len(vectors[index_to_word[0]])
    w = np.zeros((vocabulary_size, vector_dim))
    for word, vector in vectors.items():
        if word == '<unk>':
            continue
        w[word_to_index[word], :] = vector

    if normalize:
        # normalize each word vector to unit length
        d = np.sum(w ** 2, 1) ** 0.5
        return (w.T / d).T
    else:
        return w


def load_vector_array(vectors_file, vocabulary_file, normalize=True):
    """
    Load GloVe word vectors into a 2D numpy array with one row per word vector.

    :param vectors_file: file returned by GloVe containing word vectors, assumed to be gzipped
    :param vocabulary_file: vocabulary file fed into GloVe containing words, assumed to be gzipped
    :param normalize: normalize word vectors to unit L2 norm
    :return: tuple of ndarray, dict, dict. The word vectors, a dict mapping each word to row index and another dict
     that maps row indices to words.
    """
    vectors = {}
    with gzip.open(vectors_file, 'rt', encoding='utf8', errors='raise') as file_in:
        for line in file_in:
            values = line.rstrip().split(' ')
            vectors[values[0]] = [float(x) for x in values[1:]]

    word_to_index, index_to_word = _load_vocabulary_dicts(vocabulary_file)
    return _get_vectors(vectors, word_to_index, index_to_word, normalize), word_to_index, index_to_word


def load_pre_trained_vector_array(pre_trained_vectors_file, normalize=True):
    """
    Load pre-trained GloVe word vectors, as available on the project website, into a 2D numpy array with one row per
    word and word vector.

    Note that the pre-trained word vector files do not come with a vocabulary file which means that the
    mappings between words and output array are loaded from the input file directly.
    :param pre_trained_vectors_file: pre-trained word vectors file, assumed to be gzipped
    :param normalize: normalize word vectors to unit L2 norm
    :return: tuple of ndarray, dict, dict. The word vectors, a dict mapping each word to row index and another dict
     that maps row indices to words.
     """
    vectors = {}
    word_to_index = {}
    with gzip.open(pre_trained_vectors_file, 'rt', encoding='utf8', errors='raise') as file_in:
        word_index = 0
        for line in file_in:
            values = line.rstrip().split(' ')
            word = values[0]
            vectors[word] = [float(x) for x in values[1:]]
            if word != '<unk>':
                word_to_index[word] = word_index
                word_index += 1
    index_to_word = {i: w for w, i in word_to_index.items()}
    return _get_vectors(vectors, word_to_index, index_to_word, normalize), word_to_index, index_to_word
