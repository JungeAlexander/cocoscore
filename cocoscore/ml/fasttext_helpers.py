import numpy as np


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


def get_fasttext_calls(train_file_path, test_file_path, param_dict, fasttext_path, probability_file_prefix, thread=1):
    """
    Generates fastText command-line calls for training a supervised model and applying it to a test dataset.

    :param train_file_path: path to the training dataset
    :param test_file_path: path to the test dataset
    :param param_dict: dictionary mapping fasttext hyperparameters to their values
    :param fasttext_path: path to the fastText executable
    :param probability_file_prefix: str, prefix for the output file with class probabilities for the test dataset
    :param thread: int, the number of threads to use
    :return triple of str - fastText calls for training, testing, quantizing, testing with the quantized model
    """
    param_dict['-thread'] = thread
    train_args = ''
    prob_file_path = probability_file_prefix
    for arg in sorted(param_dict.keys()):
        val = param_dict[arg]
        train_args += arg + ' ' + str(val) + ' '
        prob_file_path += arg.replace('-', '') + '_' + str(val) + '_'
    prob_file_path += '.txt.gz'

    train_call = fasttext_path + ' supervised -input ' + train_file_path + ' -output model ' + train_args
    predict_call_prefix = test_file_path + ' 2 | gzip > ' + prob_file_path
    predict_call = fasttext_path + ' predict-prob model.bin ' + predict_call_prefix
    compress_call = fasttext_path + ' quantize -input model -output model'
    compress_predict_call = fasttext_path + ' predict-prob model.ftz ' + predict_call_prefix
    return train_call, predict_call, compress_call, compress_predict_call


def run_fasttext(train_file_path, test_file_path, param_dict, fasttext_path, probability_file_prefix, thread=1,
                 ):
    """
    Generates fastText command-line calls for training a supervised model and applying it to a test dataset.

    :param train_file_path: path to the training dataset
    :param test_file_path: path to the test dataset
    :param param_dict: dictionary mapping fasttext hyperparameters to their values
    :param fasttext_path: path to the fastText executable
    :param probability_file_prefix: str, prefix for the output file with class probabilities for the test dataset
    :param thread: int, the number of threads to use:
    """
    pass
