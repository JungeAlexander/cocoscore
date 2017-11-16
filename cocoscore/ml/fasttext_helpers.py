from .cv import compute_cv_fold_stats, cv_independent_associations
from ..tools.file_tools import get_file_handle
from gensim import utils
import gzip
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from statistics import mean, stdev

def get_uniform(low, high, random_seed):
    random_state = np.random.RandomState(random_seed)
    return lambda: float('%.3g' % random_state.uniform(low, high))


def get_uniform_int(low, high, random_seed):
    random_state = np.random.RandomState(random_seed)
    return lambda: random_state.randint(low, high)


def get_log_uniform(low, high, random_seed):
    random_state = np.random.RandomState(random_seed)
    return lambda: float('%.3g' % np.power(10, random_state.uniform(low, high)))


def get_discrete_uniform(values, random_seed):
    random_state = np.random.RandomState(random_seed)
    return lambda: values[random_state.randint(len(values))]


def get_hyperparameter_distributions():
    param_dict = {
        '-lr': get_log_uniform(-3, 1, 0),
        '-epoch': get_uniform_int(10, 51, 12),
        '-wordNgrams': get_uniform_int(1, 6, 23),
        '-dim': get_uniform_int(50, 500, 42),
        '-ws': get_uniform_int(3, 10, 55)
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
    param_dict['-thread'] = thread
    train_args = []
    for arg in sorted(param_dict.keys()):
        val = param_dict[arg]
        train_args += [arg, str(val)]
    train_call = [fasttext_path, 'supervised',  '-input', train_file_path, '-output', model_path]
    train_call += train_args
    if pretrained_vectors_path is not None:
        train_call += ['-pretrainedVectors', pretrained_vectors_path]
    compress_call = [fasttext_path, 'quantize', '-input',  model_path, '-output', model_path]
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
    utils.check_output(args=train_call)
    if compress_model:
        utils.check_output(args=compress_call)
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
    predictions = utils.check_output(args=predict_call)
    with gzip.open(probability_file_path, 'wb') as fout:
        fout.write(predictions)


def _compute_auroc(dataset_file_path, prob_file_path):
    labels = load_labels(dataset_file_path)
    predicted = load_fasttext_class_probabilities(prob_file_path)
    roc = roc_auc_score(labels, predicted)
    return roc


def fasttext_cv_independent_associations(data_df, param_dict, fasttext_path, cv_folds=5,
                                         entity_columns=('entity1', 'entity2'), random_state=None,
                                         thread=1, compress_model=False,
                                         pretrained_vectors_path=None):
    """
    A wrapper around `cv_independent_associations()` in `ml/cv.py` that runs fastText on each CV fold and returns
    training and validation AUROC for each fold, means and standard variation across folds along with various other
    statistics.

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
    :return: a pandas DataFrame with cross_validation results
    """
    cv_data_df = data_df.copy()  # copy needed beceause labels column needs to be changed later
    cv_sets = list(cv_independent_associations(cv_data_df, cv_folds=cv_folds, random_state=random_state,
                                               entity_columns=entity_columns))
    cv_stats_df = compute_cv_fold_stats(cv_data_df, cv_sets)
    cv_data_df['class'] = cv_data_df['class'].apply(lambda c: '__label__' + str(c))

    # write temporary files for each CV train and test fold
    cv_train_test_file_pairs = []
    for cv_iter, train_test_indices in enumerate(cv_sets):
        train_indices, test_indices = train_test_indices

        train_df = cv_data_df.iloc[train_indices, :].loc[:, ['class', 'sentence_text']]
        test_df = cv_data_df.iloc[test_indices, :].loc[:, ['class', 'sentence_text']]

        # manual printing to file as to_csv complains about space as separator and spaces within sentences
        train_path = 'cv_train_' + str(cv_iter) + '.txt'
        test_path = 'cv_test_' + str(cv_iter) + '.txt'
        for curr_file, curr_df in zip([train_path, test_path],
                                      [train_df, test_df]):
            with open(curr_file, 'wt') as fout:
                for row in curr_df.itertuples():
                    fout.write(str(row[1]) + ' ' + str(row[2]) + os.linesep)
        cv_train_test_file_pairs.append((train_path, test_path))

    # run fasttext and compute AUROC on each fold
    train_rocs = []
    test_rocs = []
    for i, train_test_path in enumerate(cv_train_test_file_pairs):
        train_file_path, test_file_path = train_test_path
        train_prob_file_path = 'train_predict-prob_iter_' + str(i) + '_'
        test_prob_file_path = 'test_predict-prob_iter_' + str(i) + '_'

        model_file = fasttext_fit(train_file_path, param_dict, fasttext_path, thread=thread,
                                  compress_model=compress_model,
                                  pretrained_vectors_path=pretrained_vectors_path)
        fasttext_predict(model_file, train_file_path, fasttext_path, train_prob_file_path)
        fasttext_predict(model_file, test_file_path, fasttext_path, test_prob_file_path)
        train_roc = _compute_auroc(train_file_path, train_prob_file_path)
        test_roc = _compute_auroc(test_file_path, test_prob_file_path)
        train_rocs.append(train_roc)
        test_rocs.append(test_roc)
        os.remove(model_file)
        os.remove(train_prob_file_path)
        os.remove(test_prob_file_path)

    # aggregate performance measures and fold statistics in result DataFrame
    results_df = pd.DataFrame()
    results_df['mean_test_score'] = [mean(test_rocs)]
    results_df['stdev_test_score'] = [stdev(test_rocs)]
    results_df['mean_train_score'] = [mean(train_rocs)]
    results_df['stdev_train_score'] = [stdev(train_rocs)]
    for stats_row in cv_stats_df.itertuples():
        cv_fold = str(stats_row.fold)
        results_df['split_' + cv_fold + '_test_score'] = [test_rocs[int(cv_fold)]]
        results_df['split_' + cv_fold + '_train_score'] = [test_rocs[int(cv_fold)]]
        results_df['split_' + cv_fold + '_n_test'] = [stats_row.n_test]
        results_df['split_' + cv_fold + '_pos_test'] = [stats_row.pos_test]
        results_df['split_' + cv_fold + '_n_train'] = [stats_row.n_train]
        results_df['split_' + cv_fold + '_pos_train'] = [stats_row.pos_train]

    # delete temporarily created CV dataset files
    for train_file, test_file in cv_train_test_file_pairs:
        os.remove(train_file)
        os.remove(test_file)

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
        true_labels = [1 if l == '__label__1' else 0 for l in true_labels]
        return true_labels
    finally:
        conn.close()
