from math import exp


def _distance_scorer(data_df, score_function):
    distance_column = 'distance'
    if distance_column not in data_df.columns:
        raise ValueError(f'The given data_df does not have a {distance_column} column.')
    distances = data_df.loc[:, distance_column]
    return distances.apply(score_function)


def reciprocal_distance(data_df, *_):
    """
    Computes reciprocal distance scores for a given DataFrame of co-mentions.

    The reciprocal distance score is defined as 1/x where x is the the distance of the closest matches of an
    entity pair of interest.

    :param data_df: pandas DataFrame, the data set loaded using
    tools.data_tools.load_data_frame(..., match_distance=True)
    :returns a pandas Series of distance scores
    """
    return polynomial_decay_distance(data_df, 1, 0, 9999999)


def constant_distance(data_df, *_):
    """
    Returns a constant distance score of 1 for a given DataFrame of co-mentions.

    :param data_df: pandas DataFrame, the data set loaded using
    tools.data_tools.load_data_frame(..., match_distance=True)
    :returns a pandas Series of distance scores
    """
    return _distance_scorer(data_df, score_function=lambda x: 1.0)


def exponential_decay_distance(data_df, k, c, m):
    """
    Computes exponentially decaying distance scores for a given DataFrame of co-mentions.

    The exponentially decaying distance score is defined as min(exp(-k*x) + c, m) where
    x is the the distance of the closest matches of an
    entity pair of interest and k, m are positive constants.

    :param data_df: pandas DataFrame, the data set loaded using
    tools.data_tools.load_data_frame(..., match_distance=True)
    :param k: float, a positive constant
    :param c: float, a positive constant
    :param m: float, a positive constant
    :returns a pandas Series of distance scores
    """
    return _distance_scorer(data_df, lambda x: min(exp(-k * max(x, 1)) + c, m))


def polynomial_decay_distance(data_df, k, c, m):
    """
    Computes polynomially decaying distance scores for a given DataFrame of co-mentions.

    The polynomially decaying distance score is defined as min(x^(-k) + c, m)where
    x is the the distance of the closest matches of an
    entity pair of interest and k, m are positive constants.

    :param data_df: pandas DataFrame, the data set loaded using
    tools.data_tools.load_data_frame(..., match_distance=True)
    :param k: float, a positive constant
    :param c: float, a positive constant
    :param m: float, a positive constant
    :returns a pandas Series of distance scores
    """
    return _distance_scorer(data_df, lambda x: min(max(x, 1) ** (-k) + c, m))
