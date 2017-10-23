#!/usr/bin/env python
import argparse
import logging
import pathlib
import string

from nltk.tokenize.stanford import StanfordTokenizer
import numpy as np

from .generate_dataset import disease_placeholder, gene_placeholder
from ..ml.feature.glove import load_vector_array, load_pre_trained_vector_array
from ..tools.data_tools import load_data_frame


__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Generate feature matrices for train/test sentences based on word vectors.
    
    Each word in the sentence is mapped to a word vector, if available, which are then
    averaged to obtain a vector representation for the entire sentence.
    ''')
    parser.add_argument('vectors_file',
                        help='Path to gzipped word vectors file as produced by GloVe. '
                             'Example: data/20170605-GloVe-vectors/run/vectors.txt.gz')
    parser.add_argument('vocabulary_file',
                        help='Path to gzipped vocabulary file as input to GloVe. Specify PRETRAINED here if '
                             'vectors_file corresponds to a pre-trained Glove file from the project website. '
                             'Example: data/20170605-GloVe-vectors/run/vocab.txt.gz')
    parser.add_argument('output_prefix',
                        help='Prefix to be added to the output feature matrices. '
                             'Example: may produce two output files in gzipped text format containing the matrices: '
                             '<prefix>_train.txt.gz and <prefix>_test.txt.gz')
    parser.add_argument('dataset_file', nargs='+',
                        help='Path to training data set file(s). '
                             'Example: results/20170507_cv/train_ghr.tsv.gz')
    parser.add_argument('--min_word_vectors_per_sentence', default=3, type=int,
                        help='Minimal number of words to be mapped to word vectors for a sentence representation '
                             'vector to be constructed. Sentences where no representation was generated are'
                             'replaced by a sequence of numpy nan.')
    parser.add_argument('--remove_punctuation_vectors', action='store_true',
                        help='If this is set, vectors representing punctuation are removed prior to computing sentence '
                             'vectors.')
    parser.add_argument('--path_to_tokenizer_jar',
                        default='/home/people/ajunge/local/stanford-parser-full-2016-10-31/stanford-parser.jar',
                        type=str,
                        help='Path to local installation of .jar file containing the Stanford tokenizer. '
                             'For installation details, see: data/20170605-GloVe-vectors/README.md')
    parser.add_argument('--normalize', action='store_true',
                        help='If this is set, word vectors are normalized to unit length before combining them to '
                             'sentence vectors.')
    args = parser.parse_args()
    return args.vectors_file, args.vocabulary_file, args.output_prefix, args.dataset_file, \
        args.min_word_vectors_per_sentence, \
        args.remove_punctuation_vectors, args.path_to_tokenizer_jar, args.normalize


def tokenize_sentences_from_str(sentences, path_to_tokenizer_jar):
    tokenizer = StanfordTokenizer(path_to_jar=path_to_tokenizer_jar)
    # concatenate all sentences using a fixed token as delimiter to only submit one call to external tokenizer
    sentence_delimiter = ' MYSENTDELIM '
    sentences_concatenated = sentence_delimiter.join(sentences)
    # split tokens into different sentences again; also lowercase tokens
    sentences_tokenized = []
    temp_sentence = []
    for word in tokenizer.tokenize(sentences_concatenated):
        if word == sentence_delimiter.strip():
            sentences_tokenized.append(temp_sentence)
            temp_sentence = []
            continue
        temp_sentence.append(word.lower())
    sentences_tokenized.append(temp_sentence)
    return sentences_tokenized


def tokenize_sentences(data_path, path_to_tokenizer_jar):
    current_df = load_data_frame(data_path)
    sentences_tokenized = tokenize_sentences_from_str(current_df.loc[:, 'sentence_text'], path_to_tokenizer_jar)
    assert len(sentences_tokenized) == len(current_df.loc[:, 'sentence_text'])
    return sentences_tokenized


def get_sentence_vector_array(sentences_tokenized, word_vectors, word_to_index, min_vector_count, remove_punctuation,
                              replace_disease_gene_tokens=True):
    # :param replace_disease_gene_tokens: boolean indicating if tokens representing tagged diseases and genes
    #         are mapped to the word vectors for 'disease' and 'gene', respectively. If False, the tokens are ignored.

    # simply average all word vectors corresponding to words in the sentence
    vector_dim = len(word_vectors[0])
    sentence_array = np.zeros((len(sentences_tokenized), vector_dim))
    for sentence_index, sentence in enumerate(sentences_tokenized):
        vector_count = 0
        sentence_vec = np.zeros(vector_dim)
        for word in sentence:
            if remove_punctuation and word.strip() in string.punctuation:
                continue
            if word not in word_to_index and replace_disease_gene_tokens and word == disease_placeholder.lower():
                vector_count += 1
                sentence_vec += word_vectors[word_to_index['disease']]
            elif word not in word_to_index and replace_disease_gene_tokens and word == gene_placeholder.lower():
                vector_count += 1
                sentence_vec += word_vectors[word_to_index['gene']]
            elif word in word_to_index:
                vector_count += 1
                sentence_vec += word_vectors[word_to_index[word]]
        if vector_count < min_vector_count:
            sentence_vec = np.full(vector_dim, np.nan)
            logging.warning('Following sentence could not be represented as a vector since only {:d} words were '
                            'mapped to vectors: {}'.format(vector_count, ' '.join(sentence)))
        sentence_array[sentence_index] = sentence_vec/vector_count
    return sentence_array


def main():
    logging.basicConfig(level=logging.INFO)
    vectors_file, vocabulary_file, output_prefix, dataset_file, min_vector_count, \
        remove_punctuation, path_to_tokenizer_jar, normalize = parse_parameters()

    if vocabulary_file == 'PRETRAINED':
        word_vectors, word_to_index, index_to_word = load_pre_trained_vector_array(vectors_file,
                                                                                   normalize=normalize)
    else:
        word_vectors, word_to_index, index_to_word = load_vector_array(vectors_file, vocabulary_file,
                                                                       normalize=normalize)

    data_output_paths = []
    for data_path in dataset_file:
        output_postfix = pathlib.Path(data_path).stem[:-len('.tsv')] + '.txt.gz'
        data_output_paths.append((data_path, output_prefix + '_' + output_postfix))

    for current_data_path, current_output_path in data_output_paths:
        sentences_tokenized = tokenize_sentences(current_data_path, path_to_tokenizer_jar)
        sentence_feature = get_sentence_vector_array(sentences_tokenized, word_vectors, word_to_index,
                                                     min_vector_count, remove_punctuation)
        np.savetxt(current_output_path, sentence_feature, fmt='%.5f')  # gzips automatically when file ending is .gz

if __name__ == '__main__':
    main()
