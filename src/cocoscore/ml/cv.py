from collections import defaultdict

import numpy as np
import pandas as pd


def _cv_independent_entities_treated_separately(data_df, cv_folds, entity_columns, random_state):
    entity1s = data_df[entity_columns[0]].unique()
    entity1s.sort()
    entity2s = data_df[entity_columns[1]].unique()
    entity2s.sort()
    random_state.shuffle(entity2s)
    random_state.shuffle(entity1s)
    for entity1s_test, entity2s_test in zip(np.array_split(entity1s, cv_folds),
                                            np.array_split(entity2s, cv_folds)):
        entity1_is_test = data_df[entity_columns[0]].isin(entity1s_test)
        entity2_is_test = data_df[entity_columns[1]].isin(entity2s_test)
        is_test_set = np.logical_and(entity1_is_test, entity2_is_test)
        test_indices = np.where(is_test_set)[0]
        is_remove_set = np.logical_xor(entity1_is_test, entity2_is_test)
        remove_indices = np.where(is_remove_set)[0]
        is_train_set = np.logical_not(np.logical_or(entity1_is_test, entity2_is_test))
        train_indices = np.where(is_train_set)[0]
        assert len(set(np.concatenate([train_indices, test_indices, remove_indices]))) == len(data_df)
        yield train_indices, test_indices


def cv_independent_entities(data_df, cv_folds=5, entity_columns=('entity1', 'entity2'),
                            treat_entity_columns_separately=False,
                            random_state=None):
    """
    Performs an entity aware splitting into CV folds to ensure independence between folds.

    Independence is established by, for instance, reserving 20% of all interacting entities for each CV test set.
    All interactions between these entities are then assigned to the given test set. The CV training
    set are the remaining associations not involving any of the previously selected entities.
    This ensures that features learned for specific entities do not bleed into the CV test sets.

    :param data_df: the DataFrame to be split up into CV folds
    :param cv_folds: int, the number of CV folds to generate
    :param entity_columns: tuple of str, column names in data_df where interacting entities can be found
    :param treat_entity_columns_separately: boolean, set this to split the given entity columns independently
    :param random_state: numpy RandomState to use while splitting into folds
    :return: a generator that returns (train_indices, test_indices) tuples where each array of indices denotes the
    indices of rows in data_df that are assigned to the train/test set in the given CV fold.
    """
    assert len(entity_columns) == 2
    if random_state is None:
        # random seeding
        random_state = np.random.RandomState()
    if treat_entity_columns_separately:
        yield from _cv_independent_entities_treated_separately(data_df, cv_folds, entity_columns, random_state)
        return
    entities = data_df[entity_columns[0]].append(data_df[entity_columns[1]]).unique()
    entities.sort()
    random_state.shuffle(entities)
    for entities_test in np.array_split(entities, cv_folds):
        first_is_test = data_df[entity_columns[0]].isin(entities_test)
        second_is_test = data_df[entity_columns[1]].isin(entities_test)
        is_test_set = np.logical_and(first_is_test, second_is_test)
        test_indices = np.where(is_test_set)[0]
        is_remove_set = np.logical_xor(first_is_test, second_is_test)
        remove_indices = np.where(is_remove_set)[0]
        is_train_set = np.logical_not(np.logical_or(first_is_test, second_is_test))
        train_indices = np.where(is_train_set)[0]
        assert len(set(np.concatenate([train_indices, test_indices, remove_indices]))) == len(data_df)
        yield train_indices, test_indices


def cv_independent_associations(data_df, cv_folds=5, entity_columns=('entity1', 'entity2'), random_state=None):
    """
    Performs an association aware splitting into CV folds to ensure independence between folds.

    Independence is established by, for instance, reserving 20% of all associations for each CV test set.
    All instances of this association are then assigned to the given test set. The CV training
    set are the remaining instances not involving any of the previously selected associations.
    This ensures that features learned for specific associations do not bleed into the CV test sets.

    :param data_df: the DataFrame to be split up into CV folds
    :param cv_folds: int, the number of CV folds to generate
    :param entity_columns: tuple of str, column names in data_df where interacting entities can be found
    :param random_state: numpy RandomState to use while splitting into folds
    :return: a generator that returns (train_indices, test_indices) tuples where each array of indices denotes the
    indices of rows in data_df that are assigned to the train/test set in the given CV fold.
    """
    assert len(entity_columns) == 2

    # interacting are first sorted and then combined to a string; this eases things later
    # because we do not need to worry about the order of entity1 and entity2 anymore
    associations = data_df.apply(lambda row: ','.join(sorted((row[entity_columns[0]], row[entity_columns[1]]))),
                                 axis=1,
                                 raw=False)
    all_assoc = sorted(set(associations))
    if random_state is None:
        # random seeding
        random_state = np.random.RandomState()
    random_state.shuffle(all_assoc)
    for entities_test in np.array_split(all_assoc, cv_folds):
        is_test_assoc = associations.isin(entities_test)
        is_train_assoc = np.logical_not(is_test_assoc)
        test_indices = np.where(is_test_assoc)[0]
        train_indices = np.where(is_train_assoc)[0]
        assert len(set(np.concatenate([train_indices, test_indices]))) == len(data_df)
        yield train_indices, test_indices


def compute_cv_fold_stats(data_df, cv_splits):
    """
    Computes CV fold size and class proportions for a given CV splitting.

    :param data_df: a DataFrame
    :param cv_splits: a given cv splitting as e.g. returned by cv_independent_associations()
    :return: a DataFrame that lists count and fraction of positives in each CV fold training and test set
    """
    folds = list(range(len(cv_splits)))
    n_train = []
    pos_train = []
    n_test = []
    pos_test = []
    for train, test in cv_splits:
        train_df = data_df.iloc[train, :]
        n_train.append(len(train_df))
        pos_train.append(train_df['class'].mean())

        test_df = data_df.iloc[test, :]
        n_test.append(len(test_df))
        pos_test.append(test_df['class'].mean())
    return pd.DataFrame({'fold': folds, 'n_train': n_train, 'pos_train': pos_train,
                         'n_test': n_test, 'pos_test': pos_test})


def _remove_missing_data_points(matrix_loaded, dataset):
    keep_rows = np.all(np.isfinite(matrix_loaded), axis=1)
    dataset_cleaned = dataset.loc[keep_rows, :]
    dataset_cleaned.reset_index(inplace=True)
    return matrix_loaded[keep_rows, :], dataset_cleaned


# def _perform_xgboost_cv(dataset, vectorizer, param_grid, scoring, iid, cv_sets, out_file, features,
#                         feature_matrix_non_missing):
#     # xgboost expects evaluation metric as parameter
#     if scoring == 'roc_auc':
#         param_grid['classifier__eval_metric'] = ['auc']
#     else:
#         raise ValueError('Unknown scoring function {} for xgboost CV.'.format(scoring))
#
#     if iid:
#         raise ValueError('CV performance for iid data is not implemented for xgboost CV.')
#
#     with open(out_file, 'wt', buffering=1, encoding='utf-8', errors='strict') as out_handle:
#         # following scikit-learn' Pipeline implementation, param_grid prefixes arguments to the
#         # classifier with 'classifier__' - this prefix has to be removed before calling xgboost
#         classifier_arg_prefix = 'classifier__'
#         param_grid_classifier = {}
#         for key, values in param_grid.items():
#             if not key.startswith(classifier_arg_prefix):
#                 raise ValueError('Cannot handle parameter {} in grid used for xgboost CV'.format(key))
#             param_grid_classifier[key[len(classifier_arg_prefix):]] = values
#
#         # following sklearn's GridSearch CV, output column names for parameters are prefixed 'param_classifier__'
#         params_header = ['param_' + classifier_arg_prefix + v for v in sorted(param_grid_classifier.keys())]
#         header = ['split', 'test_score', 'train_score'] + params_header
#         out_handle.write('\t'.join(header) + os.linesep)
#         for cv_iter, train_test_indices in enumerate(cv_sets):
#             train_indices, test_indices = train_test_indices
#             if features == 'bow':
#                 train_df = dataset.iloc[train_indices, :]
#                 test_df = dataset.iloc[test_indices, :]
#
#                 vectorizer.fit(train_df['text'])
#                 train_mat = vectorizer.transform(train_df['text'])
#                 test_mat = vectorizer.transform(test_df['text'])
#
#                 # FIXME converting sparse document-term matrix to dense is wasteful in terms of memory but
#                 #       not doing this conversion leads to errors with unknown feature names in xgb.train() below
#                 train_data = xgb.DMatrix(train_mat.todense(), train_df['class'])
#                 test_data = xgb.DMatrix(test_mat.todense(), test_df['class'])
#             elif features == 'word_vectors':
#                 train_matrix = feature_matrix_non_missing[train_indices, :]
#                 test_matrix = feature_matrix_non_missing[test_indices, :]
#                 train_labels = dataset['class'].iloc[train_indices]
#                 test_labels = dataset['class'].iloc[test_indices]
#                 train_data = xgb.DMatrix(train_matrix, train_labels)
#                 test_data = xgb.DMatrix(test_matrix, test_labels)
#             else:
#                 raise ValueError('Unknown features parameter: {}'.format(features))
#
#             for current_grid_point in ParameterGrid(param_grid_classifier):
#                 # num_round must be given to fit method directly
#                 if 'num_round' in current_grid_point:
#                     num_round = current_grid_point['num_round']
#                     del current_grid_point['num_round']
#                 else:
#                     num_round = 10  # the default in xgboost
#                 watchlist = [(test_data, 'eval'), (train_data, 'train')]
#                 eval_result = {}
#                 _ = xgb.train(current_grid_point, train_data, num_round, watchlist, evals_result=eval_result)
#                 train_score = eval_result['train'][current_grid_point['eval_metric']][-1]
#                 test_score = eval_result['eval'][current_grid_point['eval_metric']][-1]
#                 current_grid_point['num_round'] = num_round
#
#                 results = [str(cv_iter), str(test_score), str(train_score)] + \
#                           [str(current_grid_point[k[len('param_' + classifier_arg_prefix):]]) for k in params_header]
#                 out_handle.write('\t'.join(results) + os.linesep)


def _pivot_group(current_group):
    pivot_group = current_group.pivot(columns='split').reorder_levels([1, 0], axis=1)
    columns_collapsed = ['split' + str(split) + '_' + score for split, score in pivot_group.columns.to_series()]
    pivot_group.columns = columns_collapsed
    return pivot_group.sum()


# def grid_search_cv(estimator, param_grid, dataset, results_file, n_jobs, cv_folds=5, random_state=None,
#                    scoring='roc_auc', features='bow', feature_matrix=None, iid=False):
#     """
#     Performs a GridSearchCV for a given estimator, parameter grid and data set.
#
#     :param estimator: str, the estimator to use
#     :param param_grid: dict, the parameter grid to use
#     :param dataset: DataFrame, the data set to be cross-validated
#     :param results_file: str, path to write (gzipped) GridSearchCV output to
#     :param n_jobs: int, the number of parallel ML jobs to run
#     :param cv_folds: int, the number of CV fold to generate using cv_independent_associations()
#     :param random_state: RandomState to use when generating CV folds
#     :param scoring: str, scoring function to compute CV errors
#     :param features: str, 'bow' for back-of-words; 'word_vectors' for numerical vectors using word embeddings
#     :param feature_matrix: str, if features == 'word_vectors', the path to file containing the matrix of word vectors
#     :param iid: boolean, indicate whether samples in given dataset are iid
#     """
#     if estimator == 'random_forest':
#         clf = RandomForestClassifier()
#     elif estimator == 'xgboost':
#         clf = None
#         pass  # handled below since xgboost's sklearn API seems incompatible with both Pipelines and auc_roc comput.
#     else:
#         raise ValueError('Unknown estimator: {}'.format(estimator))
#
#     if features == 'word_vectors':
#         if not os.path.isfile(feature_matrix):
#             raise ValueError('When using word_vectors as features, a valid path to the feature matrix must be given.')
#     elif features == 'bow':
#         pass
#     else:
#         raise ValueError('Unknown features parameter: {}'.format(features))
#
#     if features == 'word_vectors':
#         matrix_loaded = np.loadtxt(feature_matrix)
#         # since feature vectors may be missing for some sentences, remove those from the dataset
#         matrix_non_missing, dataset = _remove_missing_data_points(matrix_loaded, dataset)
#     else:
#         matrix_non_missing = None
#
#     cv_sets = list(cv_independent_entities(dataset, cv_folds=cv_folds, random_state=random_state))
#
#     # stop_words = [gene_placeholder, disease_placeholder]
#     # vectorizer = CountVectorizer(stop_words=stop_words)
#     vectorizer = CountVectorizer()
#     if estimator != 'xgboost':
#         if features == 'bow':
#             text_clf = Pipeline([('vectorizer', vectorizer),  # ('tfidf', TfidfTransformer()),
#                                  ('classifier', clf)])
#             text_grid = GridSearchCV(text_clf, param_grid, scoring=scoring, n_jobs=n_jobs, iid=iid, cv=cv_sets)
#             _ = text_grid.fit(dataset['text'], dataset['class'])
#         elif features == 'word_vectors':
#             text_grid = GridSearchCV(clf, param_grid, scoring=scoring, n_jobs=n_jobs, iid=iid, cv=cv_sets)
#             _ = text_grid.fit(matrix_non_missing, dataset['class'])
#         else:
#             raise ValueError('Unknown features parameter: {}'.format(features))
#
#         results_df = pd.DataFrame(text_grid.cv_results_)
#     else:
#         out_file = 'tmp_xgboost.tsv'
#         _perform_xgboost_cv(dataset, vectorizer, param_grid, scoring, iid, cv_sets, out_file, features,
#                             matrix_non_missing)
#         xgboost_results = pd.DataFrame.from_csv(out_file, sep='\t', index_col=None)
#         param_columns = list(xgboost_results.filter(regex='^param_').columns)
#         grouped_results = xgboost_results.groupby(param_columns)
#         results_df = grouped_results.apply(_pivot_group).reset_index()
#
#         cv_test_score_columns = list(results_df.filter(regex='_test_score$'))
#         results_df['mean_test_score'] = results_df.loc[:, cv_test_score_columns].mean(axis=1, skipna=False)
#         results_df['std_test_score'] = results_df.loc[:, cv_test_score_columns].std(axis=1, skipna=False)
#
#         cv_train_score_columns = list(results_df.filter(regex='_train_score$'))
#         results_df['mean_train_score'] = results_df.loc[:, cv_train_score_columns].mean(axis=1, skipna=False)
#         results_df['std_train_score'] = results_df.loc[:, cv_train_score_columns].std(axis=1, skipna=False)
#
#         results_df['rank_test_score'] = results_df['mean_test_score'].rank(method='first', ascending=False)
#
#     # add CV fold stats to output data frame
#     fit_count = len(results_df)
#     cv_stats_df = compute_cv_fold_stats(dataset, cv_sets)
#     for stats_row in cv_stats_df.itertuples():
#         cv_fold = str(stats_row.fold)
#         results_df['split_' + cv_fold + '_n_test'] = [stats_row.n_test] * fit_count
#         results_df['split_' + cv_fold + '_pos_test'] = [stats_row.pos_test] * fit_count
#         results_df['split_' + cv_fold + '_n_train'] = [stats_row.n_train] * fit_count
#         results_df['split_' + cv_fold + '_pos_train'] = [stats_row.pos_train] * fit_count
#
#     with gzip.open(results_file, 'wt', encoding='utf-8', errors='raise') as file_out:
#         results_df.to_csv(file_out, sep='\t', header=True, index=False)


def get_random_parameter_sampler(param_distributions, n_iter):
    """
    Sample parameters from given distributions.

    :param param_distributions : dict
        Dictionary where the keys are parameters and values
        are distributions from which a parameter is to be sampled by calling the disribution.
    :param n_iter : integer
        Number of parameter settings that are produced.
    :return: dict of string to any
        Yields dictionaries mapping each estimator parameter to
        as sampled value.
    """
    # Code below is a simplified version of sklearn.model_selection.ParameterSampler
    # Sort the keys for reproducibility
    items = sorted(param_distributions.items())
    for _ in range(n_iter):
        params = dict()
        for k, v in items:
                params[k] = v()
        yield params


def random_cv(dataset, cv_function, cv_iterations, param_dict, param_distribution, random_seed):
    """
    Performs a cross-validation over randomly sampled parameter values for a given dataset and
    a given cross-validation function.

    :param dataset: DataFrame, the data set to be cross-validated
    :param cv_function: function that takes three arguments:
           the dataset as a pandas DataFrame, a parameter dictionary, and a numpy RandomState
           to seed the cross-validation. The function is then to perform a single cross-validation run
           and returns cross-validation results for the given parameter settings.
    :param cv_iterations: int, the number of random parameter assignments to try out
    :param param_dict: dict specifying parameters and values that are to be kept fixed
    :param param_distribution: dict mapping parameters to distributions to sample parameters from
    :param random_seed: int to seed numpy RandomState to use while splitting into CV folds in each iteration
    :return: a pandas DataFrame containing results aggregated over the CV iterations; column names should be explanatory
    """
    cv_parameters = defaultdict(list)
    cv_results = []
    for params in get_random_parameter_sampler(param_distribution, cv_iterations):
        # set parameters to be kept fixed
        for param, value in param_dict.items():
            params[param] = value

        iteration_results = cv_function(dataset, params, np.random.RandomState(random_seed))

        # save parameter settings and CV results to aggregate them later
        for param, value in params.items():
            cv_parameters[param.replace('-', '')].append(value)
        cv_results.append(iteration_results)
    # Aggregate parameter settings and CV results into output DataFrame
    results_df = pd.DataFrame(cv_parameters, columns=sorted(cv_parameters.keys()))
    cv_iteration_results = pd.concat(cv_results)
    results_df = pd.concat([results_df.reset_index(drop=True), cv_iteration_results.reset_index(drop=True)], axis=1)
    return results_df
