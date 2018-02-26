def _distance_scorer(data_df, score_function):
    distance_column = 'distance'
    if distance_column not in data_df.columns:
        raise ValueError(f'The given data_df does not have a {distance_column} column.')
    distances = data_df.loc[:, distance_column]
    return distances.apply(score_function)


def reciprocal_distance(data_df):
    """
    Computes reciprocal distance scores for a given DataFrame of co-mentions.

    The reciprocal distance score is defined as 1/x where x is the the distance of the closest matches of an
    entity pair of interest.

    :param data_df: pandas DataFrame, the data set loaded using
    tools.data_tools.load_data_frame(..., match_distance=True)
    :returns a pandas Series of distance scores
    """
    return _distance_scorer(data_df, score_function=lambda x: 1/x)


def constant_distance(data_df):
    """
    Returns a constant distance score of 1 for a given DataFrame of co-mentions.

    :param data_df: pandas DataFrame, the data set loaded using
    tools.data_tools.load_data_frame(..., match_distance=True)
    :returns a pandas Series of distance scores
    """
    return _distance_scorer(data_df, score_function=lambda x: 1.0)
