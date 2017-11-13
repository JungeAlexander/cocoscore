from ..tools.file_tools import get_file_handle
from gensim import utils
import gzip
import numpy as np
import os


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


def get_fasttext_train_calls(train_file_path, param_dict, fasttext_path, model_path, thread=1):
    """
    Generates fastText command-line calls for training a supervised model and for compressing the output model.

    :param train_file_path: path to the training dataset
    :param param_dict: dictionary mapping fasttext hyperparameters to their values
    :param fasttext_path: path to the fastText executable
    :param model_path: str, path to output model
    :param thread: int, the number of threads to use
    :return tuple of str - fastText calls for training and quantizing
    """
    param_dict['-thread'] = thread
    train_args = []
    for arg in sorted(param_dict.keys()):
        val = param_dict[arg]
        train_args += [arg, str(val)]
    train_call = [fasttext_path, 'supervised',  '-input', train_file_path, '-output', model_path]
    train_call += train_args
    compress_call = [fasttext_path, 'quantize', '-input',  model_path, '-output', model_path]
    return train_call, compress_call


def fasttext_fit(train_file_path, param_dict, fasttext_path, thread=1, compress_model=False, model_path='model',
                 pretrained_vectos=None):
    """
    Trains a fastText supervised model. This is a wrapper around the fastText command line interface.

    :param train_file_path: path to the training dataset
    :param param_dict: dictionary mapping fasttext hyperparameters to their values
    :param fasttext_path: path to the fastText executable
    :param thread: int, the number of threads to use
    :param compress_model: indicates whether the fastText model should be compressed (using fastText's quantize).
    :param model_path: str, path to output model
    :return str: path to trained model
    """
    train_call, compress_call = get_fasttext_train_calls(train_file_path, param_dict, fasttext_path, model_path, thread)
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
