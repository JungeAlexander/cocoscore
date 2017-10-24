import pandas as pd


def load_cv_results(results_path):
    """
    
    :param results_path: path to results file as produced by dicomclass.ml.cv.grid_search_cv()
    :return: pandas DataFrame loaded from the given path
    """
    return pd.DataFrame.from_csv(results_path, sep='\t', index_col=None, encoding='utf-8')
