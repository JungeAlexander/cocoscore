import pandas as pd


def load_data_frame(data_frame_path, sort_reindex=False, class_labels=True, match_distance=False,
                    allow_missing_text=False, quoting=3):
    """
    Load a dataset as pandas DataFrame from a given path.

    :param data_frame_path: the path to load the pandas DataFrame from
    :param sort_reindex: if True, the returned data frame will be sorted by PMID and reindex by 0, 1, 2, ...
    :param class_labels: if True, the class label is given after the mandatory text column
    :param match_distance: if True, the distance between the closest match is assumed to be present as the last column
    :param allow_missing_text: if True, missing text is replaced with empty strings
    :param quoting: int, controls field quoting in pandas.read_csv():
    "Use one of QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or QUOTE_NONE (3)."
    See also: https://docs.python.org/3/library/csv.html#csv.QUOTE_NONE
    :return: a pandas DataFrame loaded from the given path
    :raises ValueError if missing data is encountered
    """
    column_names = ['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'text']
    if class_labels:
        column_names.append('class')
    if match_distance:
        column_names.append('distance')
    data_df = pd.read_csv(data_frame_path, sep='\t', header=None, index_col=False,
                          names=column_names, quoting=quoting)
    if sort_reindex:
        data_df.sort_values('pmid', axis=0, inplace=True, kind='mergesort')
        data_df.reset_index(inplace=True, drop=True)
    if allow_missing_text:
        data_df['text'] = data_df['text'].fillna('')
    if data_df.isnull().sum().sum() != 0:
        raise ValueError(f'Encountered missing values while loading data from {data_frame_path}.')
    return data_df


def fill_missing_paragraph_document_scores(data):
    """
    Fill missing paragraph- and document-level scores for all sentence-level matches and fill all
    missing document-level scores for all document-level matches.

    Filled matches receive a score of 0.
    :param data: the scored DataFrame to fill
    :return: a new DataFrame with missing scores filled in
    """
    text = ''
    distance = -1
    predicted = 0
    filled_dfs = []
    for _, group_df in data.groupby(['entity1', 'entity2']):
        group_df = group_df.copy()
        if group_df.ndim == 1:
            group_df = group_df.to_frame()
        entity1 = group_df.loc[:, 'entity1'].iloc[0]
        entity2 = group_df.loc[:, 'entity2'].iloc[0]
        current_class = group_df.loc[:, 'class'].iloc[0]

        sentence_level = {tuple(x)
                          for x in group_df.loc[:, ['pmid', 'paragraph', 'sentence']].to_records(index=False)
                          if x[1] != -1 and x[2] != -1}
        paragraph_level = {tuple(x)
                           for x in group_df.loc[:, ['pmid', 'paragraph', 'sentence']].to_records(index=False)
                           if x[1] != -1 and x[2] == -1}
        document_level = {tuple(x)
                          for x in group_df.loc[:, ['pmid', 'paragraph', 'sentence']].to_records(index=False)
                          if x[1] == -1 and x[2] == -1}

        add_tuples = set()
        for sentence in sentence_level:
            expected_paragraph = (sentence[0], sentence[1], -1)
            expected_document = (sentence[0], -1, -1)
            if expected_paragraph not in paragraph_level:
                add_tuples.add(expected_paragraph)
            if expected_document not in document_level:
                add_tuples.add(expected_document)
        for paragraph in paragraph_level:
            expected_document = (paragraph[0], -1, -1)
            if expected_document not in document_level:
                add_tuples.add(expected_document)

        new_rows = []
        for add_tuple in add_tuples:
            new_rows.append(list(add_tuple) + [entity1, entity2, text, current_class, distance, predicted])
        if len(new_rows) > 1:
            new_df = pd.DataFrame.from_records(new_rows)
            new_df.columns = group_df.columns
            new_df = pd.concat([group_df, new_df], axis=0, ignore_index=True)
            filled_dfs.append(new_df)
        else:
            filled_dfs.append(group_df)
    filled_df = pd.concat(filled_dfs, axis=0, ignore_index=True)
    filled_df.sort_values(['pmid', 'paragraph', 'sentence'], axis=0, inplace=True, kind='mergesort')
    filled_df.reset_index(inplace=True, drop=True)
    return filled_df
