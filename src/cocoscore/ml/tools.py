import numpy as np
import pandas as pd


def load_cv_results(results_path):
    """

    :param results_path: path to results file as produced by dicomclass.ml.cv.grid_search_cv()
    :return: pandas DataFrame loaded from the given path
    """
    return pd.DataFrame.from_csv(results_path, sep='\t', index_col=None, encoding='utf-8')


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
