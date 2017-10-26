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
    train_args = ''
    for arg in sorted(param_dict.keys()):
        val = param_dict[arg]
        train_args += arg + ' ' + str(val) + ' '
    train_call = fasttext_path + ' supervised -input ' + train_file_path + ' -output ' + model_path + ' ' + train_args
    compress_call = fasttext_path + ' quantize -input ' + model_path + ' -output ' + model_path
    return train_call, compress_call


def fasttext_fit(train_file_path, param_dict, fasttext_path, thread=1, compress_model=False, model_path='model'):
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
    model_file = 'model'
    model_path = model_file + '.bin'
    # remove auxiliary vectors file
    os.remove(model_file + '.vec')
    # remove non-compressed model file if compression was performed
    if compress_model:
        os.remove(model_path)
        model_path = model_file + '.ftz'
    return model_path


def get_fasttext_test_calls(test_file_path, fasttext_path, model_path, probability_file_path):
    """
    Generates fastText command-line calls to apply a previously trained model to a test dataset. Note, this only
    supports binary classification scenarios.

    :param test_file_path: path to the test dataset
    :param fasttext_path: path to the fastText executable
    :param model_path: str, path to output model
    :param probability_file_path: str, path to the output file with class probabilities for the test dataset
    :return str - fastText calls for testing
    """
    class_count = 2
    predict_call = fasttext_path + ' predict-prob ' + model_path + ' ' + test_file_path + ' ' + str(class_count) + \
        ' | gzip > ' + probability_file_path
    return predict_call


def fasttext_predict(trained_model_path, test_file_path, fasttext_path, probability_file_path):
    """
    Predicts class probabilities for a given dataset using a previously trained fastText model.

    :param trained_model_path: path to the trained fastText model
    :param test_file_path: path to the test dataset
    :param fasttext_path: path to the fastText executable
    :param probability_file_path: str, path to the output file with class probabilities for the test dataset
    """
    predict_call = get_fasttext_test_calls(test_file_path, fasttext_path, trained_model_path, probability_file_path)
    utils.check_output(args=predict_call)


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
