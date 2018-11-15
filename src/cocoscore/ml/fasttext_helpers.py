import gzip
import os
import shutil
import subprocess
from statistics import mean
from statistics import stdev

import numpy as np
import pandas as pd
from gensim import utils
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from ..tools.file_tools import get_file_handle
from .cv import compute_cv_fold_stats
from .cv import cv_independent_associations
from .tools import get_log_uniform
from .tools import get_uniform_int


def get_hyperparameter_distributions(random_seed=None):
    """
    :param random_seed: int to seed numpy RandomState to use while initiating parameter distributions to sample from
    :return: a dictionary mapping the most important fastText parameters for classification to distributions
    to sample parameters from.
    """
    if random_seed is None:
        seeds = [0, 12, 23, 42, 55]
    else:
        random_state = np.random.RandomState(random_seed)
        seeds = random_state.randint(100000, size=5)
    param_dict = {
        '-lr': get_log_uniform(-3, 1, seeds[0]),
        '-epoch': get_uniform_int(10, 51, seeds[1]),
        '-wordNgrams': get_uniform_int(1, 6, seeds[2]),
        '-dim': get_uniform_int(50, 500, seeds[3]),
        '-ws': get_uniform_int(3, 10, seeds[4])
    }
    return param_dict


def get_fasttext_train_calls(train_file_path, param_dict, fasttext_path, model_path, thread=1,
                             pretrained_vectors_path=None):
    """
    Generates fastText command-line calls for training a supervised model and for compressing the output model.

    :param train_file_path: path to the training dataset
    :param param_dict: dictionary mapping fasttext hyperparameters to their values
    :param fasttext_path: path to the fastText executable
    :param model_path: str, path to output model
    :param thread: int, the number of threads to use
    :param pretrained_vectors_path: str, path to pre-trained `.vec` file with word embeddings
    :return tuple of str - fastText calls for training and quantizing
    """
    train_args = []
    for arg in sorted(param_dict.keys()):
        val = param_dict[arg]
        train_args += [arg, str(val)]
    train_call = [fasttext_path, 'supervised', '-input', train_file_path, '-output', model_path]
    train_call += train_args
    train_call += ['-thread', str(thread)]
    if pretrained_vectors_path is not None:
        train_call += ['-pretrainedVectors', pretrained_vectors_path]
    compress_call = [fasttext_path, 'quantize', '-input', model_path, '-output', model_path]
    return train_call, compress_call


def fasttext_fit(train_file_path, param_dict, fasttext_path, thread=1, compress_model=False, model_path='model',
                 pretrained_vectors_path=None):
    """
    Trains a fastText supervised model. This is a wrapper around the fastText command line interface.

    :param train_file_path: path to the training dataset
    :param param_dict: dictionary mapping fasttext hyperparameters to their values
    :param fasttext_path: path to the fastText executable
    :param thread: int, the number of threads to use
    :param compress_model: indicates whether the fastText model should be compressed (using fastText's quantize).
    :param model_path: str, path to output model
    :param pretrained_vectors_path: str, path to pre-trained `.vec` file with word embeddings
    :return str: path to trained model
    """
    train_call, compress_call = get_fasttext_train_calls(train_file_path, param_dict, fasttext_path, model_path, thread,
                                                         pretrained_vectors_path=pretrained_vectors_path)
    utils.check_output(args=train_call, stderr=subprocess.DEVNULL)
    if compress_model:
        utils.check_output(args=compress_call, stderr=subprocess.DEVNULL)
    model_file = model_path + '.bin'
    # remove auxiliary vectors file
    os.remove(model_path + '.vec')
    # remove non-compressed model file if compression was performed
    if compress_model:
        os.remove(model_file)
        model_file = model_path + '.ftz'
    return model_file


def get_fasttext_test_calls(test_file_path, fasttext_path, model_path):
    """
    Generates fastText command-line calls to apply a previously trained model to a test dataset. Note, this only
    supports binary classification scenarios.

    :param test_file_path: path to the test dataset
    :param fasttext_path: path to the fastText executable
    :param model_path: str, path to output model
    :return str - fastText calls for testing
    """
    class_count = 2
    predict_call = [fasttext_path, 'predict-prob', model_path, test_file_path, str(class_count)]
    return predict_call


def fasttext_predict(trained_model_path, test_file_path, fasttext_path, probability_file_path):
    """
    Predicts class probabilities for a given dataset using a previously trained fastText model.

    :param trained_model_path: path to the trained fastText model
    :param test_file_path: path to the test dataset
    :param fasttext_path: path to the fastText executable
    :param probability_file_path: str, path to the output file with class probabilities for the test dataset;
        output written to this file will always be gzipped
    """
    predict_call = get_fasttext_test_calls(test_file_path, fasttext_path, trained_model_path)
    predictions = utils.check_output(args=predict_call, stderr=subprocess.DEVNULL)
    with gzip.open(probability_file_path, 'wb') as fout:
        fout.write(predictions)


def fasttext_fit_predict_default(train_df, test_df, dim=300, epochs=50, lr=0.005, wordngrams=2, ws=5,
                                 bucket=2000000, pretrained_vectors_path=None, thread=1, output_model_path=None,
                                 output_sentence_score_path=None,
                                 shuffle=True):
    """
    Fit and predict fastText with default parameters for given training and test set.

    :param train_df: Training dataframe as returned by tools.data_tools.load_data_frame()
    :param test_df: Test dataframe as returned by tools.data_tools.load_data_frame()
    :param dim: fasttext parameter
    :param epochs: fasttext parameter
    :param lr: fasttext parameter
    :param wordngrams: fasttext parameter
    :param ws: fasttext parameter
    :param bucket: fasttext parameter
    :param pretrained_vectors_path: pretrained word embeddings
    :param thread: int, the number of threads to use by fastText
    :param output_model_path: str, path to save the fitted model to. If None, the model is not saved.
    :param output_sentence_score_path: str, path to save the test set sentence scores to, file will be gzipped. If None,
    the sentence scores are not saved.
    :param shuffle: boolean, whether the texts should be shuffled prior to prediction
    :return: tuple of training and test performance
    """
    fasttext_path = 'fasttext'
    compress_model = True
    param_dict = {'-dim': dim, '-epoch': epochs, '-lr': lr, '-wordNgrams': wordngrams, '-ws': ws, '-bucket': bucket}
    return _fasttext_fit_predict(train_df['text'].str.lower(),
                                 get_fasttext_classes(train_df),
                                 test_df['text'].str.lower(),
                                 get_fasttext_classes(test_df),
                                 param_dict,
                                 fasttext_path,
                                 thread,
                                 compress_model,
                                 pretrained_vectors_path,
                                 output_model_path=output_model_path,
                                 output_sentence_score_path=output_sentence_score_path,
                                 shuffle=shuffle)


def _fasttext_fit_predict(train_text_series, train_class_series,
                          test_text_series, test_class_series,
                          param_dict, fasttext_path, thread, compress_model,
                          pretrained_vectors_path,
                          metric='roc_auc_score',
                          output_model_path=None,
                          output_sentence_score_path=None,
                          shuffle=False):
    if shuffle:
        p_train = np.random.permutation(len(train_text_series))
        train_text_series = train_text_series.iloc[p_train]
        train_class_series = train_class_series.iloc[p_train]
        p_test = np.random.permutation(len(test_text_series))
        test_text_series = test_text_series.iloc[p_test]
        test_class_series = test_class_series.iloc[p_test]
    else:
        p_train = None
        p_test = None

    # manual printing to file as to_csv complains about space as separator and spaces within sentences
    train_path = 'tmp_ft_train.txt'
    test_path = 'tmp_ft_test.txt'
    for curr_file, text_class in zip([train_path, test_path],
                                     [(train_text_series, train_class_series),
                                      (test_text_series, test_class_series)]):
        with open(curr_file, 'wt', encoding='utf-8') as fout:
            for text, _class in zip(*text_class):
                fout.write(str(_class) + ' ' + str(text) + os.linesep)

    train_prob_file_path = 'tmp_ft_train-prob_iter_'
    test_prob_file_path = 'tmp_ft_test-prob_iter_'
    try:
        model_file = fasttext_fit(train_path, param_dict, fasttext_path, thread=thread,
                                  compress_model=compress_model,
                                  pretrained_vectors_path=pretrained_vectors_path)
        fasttext_predict(model_file, train_path, fasttext_path, train_prob_file_path)
        fasttext_predict(model_file, test_path, fasttext_path, test_prob_file_path)
        if output_model_path is not None:
            shutil.move(model_file, output_model_path)
        else:
            os.remove(model_file)
        train_metric, train_scores = _compute_metric(train_path, train_prob_file_path, metric=metric)
        test_metric, test_scores = _compute_metric(test_path, test_prob_file_path, metric=metric)
        if shuffle:
            p_train_inv = np.argsort(p_train)
            train_scores = [train_scores[i] for i in p_train_inv]

            p_test_inv = np.argsort(p_test)
            test_scores = [test_scores[i] for i in p_test_inv]

    except subprocess.CalledProcessError:
        # fastText may fail (e.g. segfault) for some parameter combinations
        raise IOError('fasttext failed in _fasttext_fit_predict.')
    finally:
        os.remove(train_path)
        os.remove(train_prob_file_path)
        if output_sentence_score_path is not None:
            with gzip.open(test_prob_file_path, "rt", encoding="utf-8") as f2,\
                open(test_path, "rt", encoding="utf-8") as f1,\
                    gzip.open(output_sentence_score_path, "wt", encoding="utf-8") as fout:
                for l1, l2 in zip(f1, f2):
                    fout.write(l1.rstrip() + "\t" + l2)
        os.remove(test_prob_file_path)
        os.remove(test_path)
    return train_metric, train_scores, test_metric, test_scores


def _compute_metric(dataset_file_path, prob_file_path, metric='roc_auc_score'):
    labels = load_labels(dataset_file_path)
    predicted = load_fasttext_class_probabilities(prob_file_path)
    if metric == 'roc_auc_score':
        score = roc_auc_score(labels, predicted)
    elif metric == 'average_precision_score':
        score = average_precision_score(labels, predicted)
    else:
        raise ValueError(f'Unknown scoring metric: {metric}')
    return score, predicted


def get_fasttext_classes(dataframe):
    return dataframe['class'].apply(lambda c: '__label__' + str(c))


def fasttext_cv_independent_associations(data_df, param_dict, fasttext_path, cv_folds=5,
                                         entity_columns=('entity1', 'entity2'), random_state=None,
                                         thread=1, compress_model=False,
                                         pretrained_vectors_path=None,
                                         metric='roc_auc_score'):
    """
    A wrapper around `cv_independent_associations()` in `ml/cv.py` that runs fastText on each CV fold and returns
    training and validation (by default) AUROC for each fold, means and standard variation across folds along with
    various other statistics.

    :param data_df: the DataFrame to be split up into CV folds
    :param param_dict: dictionary mapping fasttext hyperparameters to their values
    :param fasttext_path: path to the fastText executable
    :param cv_folds: int, the number of CV folds to generate
    :param entity_columns: tuple of str, column names in data_df where interacting entities can be found
    :param random_state: numpy RandomState to use while splitting into folds
    :param thread: int, the number of threads to use by fastText
    :param compress_model: indicates whether the fastText model should be compressed (using fastText's quantize) after
                           training.
    :param pretrained_vectors_path: str, path to pre-trained `.vec` file with word embeddings
    :param metric: performance metric used for evaluation - can be either 'roc_auc_score' (the default) or
    'average_precision_score'
    :return: a pandas DataFrame with cross_validation results
    """
    cv_sets = list(cv_independent_associations(data_df, cv_folds=cv_folds, random_state=random_state,
                                               entity_columns=entity_columns))
    cv_stats_df = compute_cv_fold_stats(data_df, cv_sets)

    # write temporary files for each CV train and test fold
    # then run fasttext and compute AUROC on each fold
    train_performances = []
    test_performances = []
    for cv_iter, train_test_indices in enumerate(cv_sets):
        train_indices, test_indices = train_test_indices

        train_df = data_df.iloc[train_indices, :]
        test_df = data_df.iloc[test_indices, :]
        try:
            train_performance, _, test_performance, _ = _fasttext_fit_predict(train_df['text'].str.lower(),
                                                                              get_fasttext_classes(train_df),
                                                                              test_df['text'].str.lower(),
                                                                              get_fasttext_classes(test_df),
                                                                              param_dict,
                                                                              fasttext_path,
                                                                              thread,
                                                                              compress_model,
                                                                              pretrained_vectors_path,
                                                                              metric=metric)
            train_performances.append(train_performance)
            test_performances.append(test_performance)
        except IOError:
            # return missing results if fasttext failed for at least one CV fold
            results_df = pd.DataFrame()
            results_df['mean_test_score'] = [np.nan]
            results_df['stdev_test_score'] = [np.nan]
            results_df['mean_train_score'] = [np.nan]
            results_df['stdev_train_score'] = [np.nan]
            for stats_row in cv_stats_df.itertuples():
                cv_fold = str(stats_row.fold)
                results_df['split_' + cv_fold + '_test_score'] = [np.nan]
                results_df['split_' + cv_fold + '_train_score'] = [np.nan]
                results_df['split_' + cv_fold + '_n_test'] = [np.nan]
                results_df['split_' + cv_fold + '_pos_test'] = [np.nan]
                results_df['split_' + cv_fold + '_n_train'] = [np.nan]
                results_df['split_' + cv_fold + '_pos_train'] = [np.nan]
            return results_df

    # aggregate performance measures and fold statistics in result DataFrame
    results_df = pd.DataFrame()
    results_df['mean_test_score'] = [mean(test_performances)]
    results_df['stdev_test_score'] = [stdev(test_performances)]
    results_df['mean_train_score'] = [mean(train_performances)]
    results_df['stdev_train_score'] = [stdev(train_performances)]
    for stats_row in cv_stats_df.itertuples():
        cv_fold = str(stats_row.fold)
        results_df['split_' + cv_fold + '_test_score'] = [test_performances[int(cv_fold)]]
        results_df['split_' + cv_fold + '_train_score'] = [test_performances[int(cv_fold)]]
        results_df['split_' + cv_fold + '_n_test'] = [stats_row.n_test]
        results_df['split_' + cv_fold + '_pos_test'] = [stats_row.pos_test]
        results_df['split_' + cv_fold + '_n_train'] = [stats_row.n_train]
        results_df['split_' + cv_fold + '_pos_train'] = [stats_row.pos_train]
    return results_df


def load_fasttext_class_probabilities(probability_file_path):
    """
    Utility function that loads class probabilities from a previously performed prediction run.

    :param probability_file_path: str, path to the output file with class probabilities for the test dataset
    :return: list of float: probability of belonging to the positive class for each example in the test dataset
    """
    probabilities = []
    with gzip.open(probability_file_path, 'rt') as fin:
        for line in fin:
            cols = line.rstrip().split()
            prob = None
            for i, col in enumerate(cols):
                if col == '__label__1':
                    prob = float(cols[i + 1])
            assert prob is not None
            probabilities.append(prob)
    return probabilities


def load_labels(dataset_path, compression=False):
    """
    Load class labels from a given dataset.
    :param dataset_path: str, path to dataset
    :param compression: boolean, indicates whether or not dataset_path is gzipped
    :return: list of 0/1, depending on class label of the instances
    """
    conn = None
    try:
        conn = get_file_handle(dataset_path, compression)
        true_labels = []
        for line in conn:
            true_labels.append(line.split()[0])
        true_labels = [1 if ll == '__label__1' else 0 for ll in true_labels]
        return true_labels
    finally:
        conn.close()
