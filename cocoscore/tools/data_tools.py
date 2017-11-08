import pandas as pd


def load_data_frame(data_frame_path, sort_reindex=False, class_labels=True):
    """
    Load a sentence data set as pandas DataFrame from a given path.

    :param data_frame_path: the path to load the pandas DataFrame from
    :param sort_reindex: if True, the returned data frame will be sorted by PMID and reindex by 0, 1, 2, ...
    :param class_labels: if True, the class label is assumed to be present as the last column
    :return: a pandas DataFrame loaded from the given path
    """
    column_names = ['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'sentence_text']
    if class_labels:
        column_names.append('class')
    data_df = pd.read_csv(data_frame_path, sep='\t', header=None, index_col=False,
                          names=column_names)
    if sort_reindex:
        data_df.sort_values('pmid', axis=0, inplace=True, kind='mergesort')
        data_df.reset_index(inplace=True, drop=True)
    assert data_df.isnull().sum().sum() == 0
    return data_df
