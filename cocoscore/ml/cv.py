import gzip
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
import xgboost as xgb

# from dicomclass.dataset.generate_dataset import disease_placeholder, gene_placeholder


def cv_independent_associations(data_df, cv_folds=5, random_state=None):
    """
    Performs a disease- and gene-aware splitting into CV folds to ensure independence between folds.
    
    Independence is established by, for instance, reserving 20% of all diseases and genes for each CV test set.
    All interactions between these diseases/genes are then assigned to the given test set. The CV training
    set are the remaining associations not involving any of the previously selected diseases/genes.
    This ensures that features learned for specific diseases/genes to not bleed into the CV test sets.
    
    :param data_df: the DataFrame to be split up into CV folds
    :param cv_folds: int, the number of CV folds to generate
    :param random_state: numpy RandomState to use while splitting into folds
    :return: a generator that returns (train_indices, test_indices) tuples where each array of indices denotes the
    indices of rows in data_df that are assigned to the train/test set in the given CV fold.
    """
    diseases = data_df['disease'].unique()
    diseases.sort()
    genes = data_df['gene'].unique()
    genes.sort()
    if random_state is None:
        # random seeding
        random_state = np.random.RandomState()
    random_state.shuffle(genes)
    random_state.shuffle(diseases)
    for diseases_test, genes_test in zip(np.array_split(diseases, cv_folds),
                                         np.array_split(genes, cv_folds)):
        disease_is_test = data_df['disease'].isin(diseases_test)
        gene_is_test = data_df['gene'].isin(genes_test)
        is_test_set = np.logical_and(disease_is_test, gene_is_test)
        test_indices = np.where(is_test_set)[0]
        is_remove_set = np.logical_xor(disease_is_test, gene_is_test)
        remove_indices = np.where(is_remove_set)[0]
        is_train_set = np.logical_not(np.logical_or(disease_is_test, gene_is_test))
        train_indices = np.where(is_train_set)[0]
        assert len(set(np.concatenate([train_indices, test_indices, remove_indices]))) == len(data_df)
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


def _perform_xgboost_cv(dataset, vectorizer, param_grid, scoring, iid, cv_sets, out_file, features,
                        feature_matrix_non_missing):
    # xgboost expects evaluation metric as parameter
    if scoring == 'roc_auc':
        param_grid['classifier__eval_metric'] = ['auc']
    else:
        raise ValueError('Unknown scoring function {} for xgboost CV.'.format(scoring))

    if iid:
        raise ValueError('CV performance for iid data is not implemented for xgboost CV.')

    with open(out_file, 'wt', buffering=1, encoding='utf-8', errors='strict') as out_handle:
        # following scikit-learn' Pipeline implementation, param_grid prefixes arguments to the
        # classifier with 'classifier__' - this prefix has to be removed before calling xgboost
        classifier_arg_prefix = 'classifier__'
        param_grid_classifier = {}
        for key, values in param_grid.items():
            if not key.startswith(classifier_arg_prefix):
                raise ValueError('Cannot handle parameter {} in grid used for xgboost CV'.format(key))
            param_grid_classifier[key[len(classifier_arg_prefix):]] = values

        # following sklearn's GridSearch CV, output column names for parameters are prefixed 'param_classifier__'
        params_header = ['param_' + classifier_arg_prefix + v for v in sorted(param_grid_classifier.keys())]
        header = ['split', 'test_score', 'train_score'] + params_header
        out_handle.write('\t'.join(header) + os.linesep)
        for cv_iter, train_test_indices in enumerate(cv_sets):
            train_indices, test_indices = train_test_indices
            if features == 'bow':
                train_df = dataset.iloc[train_indices, :]
                test_df = dataset.iloc[test_indices, :]

                vectorizer.fit(train_df['sentence_text'])
                train_mat = vectorizer.transform(train_df['sentence_text'])
                test_mat = vectorizer.transform(test_df['sentence_text'])

                # FIXME converting sparse document-term matrix to dense is wasteful in terms of memory but
                #       not doing this conversion leads to errors with unknown feature names in xgb.train() below
                train_data = xgb.DMatrix(train_mat.todense(), train_df['class'])
                test_data = xgb.DMatrix(test_mat.todense(), test_df['class'])
            elif features == 'word_vectors':
                train_matrix = feature_matrix_non_missing[train_indices, :]
                test_matrix = feature_matrix_non_missing[test_indices, :]
                train_labels = dataset['class'].iloc[train_indices]
                test_labels = dataset['class'].iloc[test_indices]
                train_data = xgb.DMatrix(train_matrix, train_labels)
                test_data = xgb.DMatrix(test_matrix, test_labels)
            else:
                raise ValueError('Unknown features parameter: {}'.format(features))

            for current_grid_point in ParameterGrid(param_grid_classifier):
                # num_round must be given to fit method directly
                if 'num_round' in current_grid_point:
                    num_round = current_grid_point['num_round']
                    del current_grid_point['num_round']
                else:
                    num_round = 10  # the default in xgboost
                watchlist = [(test_data, 'eval'), (train_data, 'train')]
                eval_result = {}
                _ = xgb.train(current_grid_point, train_data, num_round, watchlist, evals_result=eval_result)
                train_score = eval_result['train'][current_grid_point['eval_metric']][-1]
                test_score = eval_result['eval'][current_grid_point['eval_metric']][-1]
                current_grid_point['num_round'] = num_round

                results = [str(cv_iter), str(test_score), str(train_score)] + \
                          [str(current_grid_point[k[len('param_' + classifier_arg_prefix):]]) for k in params_header]
                out_handle.write('\t'.join(results) + os.linesep)


def _pivot_group(current_group):
    pivot_group = current_group.pivot(columns='split').reorder_levels([1, 0], axis=1)
    columns_collapsed = ['split' + str(split) + '_' + score for split, score in pivot_group.columns.to_series()]
    pivot_group.columns = columns_collapsed
    return pivot_group.sum()


def grid_search_cv(estimator, param_grid, dataset, results_file, n_jobs, cv_folds=5, random_state=None,
                   scoring='roc_auc', features='bow', feature_matrix=None, iid=False):
    """
    Performs a GridSearchCV for a given estimator, parameter grid and data set.
    
    :param estimator: str, the estimator to use
    :param param_grid: dict, the parameter grid to use 
    :param dataset: DataFrame, the data set to be cross-validated
    :param results_file: str, path to write (gzipped) GridSearchCV output to
    :param n_jobs: int, the number of parallel ML jobs to run
    :param cv_folds: int, the number of CV fold to generate using cv_independent_associations()
    :param random_state: RandomState to use when generating CV folds
    :param scoring: str, scoring function to compute CV errors
    :param features: str, 'bow' for back-of-words; 'word_vectors' for numerical vectors using word embeddings
    :param feature_matrix: str, if features == 'word_vectors', the path to file containing the matrix of word vectors
    :param iid: boolean, indicate whether samples in given dataset are iid
    """
    if estimator == 'random_forest':
        clf = RandomForestClassifier()
    elif estimator == 'xgboost':
        clf = None
        pass  # handled below since xgboost's sklearn API seems incompatible with both Pipelines and auc_roc computation
    else:
        raise ValueError('Unknown estimator: {}'.format(estimator))

    if features == 'word_vectors':
        if not os.path.isfile(feature_matrix):
            raise ValueError('When using word_vectors as features, a valid path to the feature matrix must be given.')
    elif features == 'bow':
        pass
    else:
        raise ValueError('Unknown features parameter: {}'.format(features))

    if features == 'word_vectors':
        matrix_loaded = np.loadtxt(feature_matrix)
        # since feature vectors may be missing for some sentences, remove those from the dataset
        matrix_non_missing, dataset = _remove_missing_data_points(matrix_loaded, dataset)
    else:
        matrix_non_missing = None

    cv_sets = list(cv_independent_associations(dataset, cv_folds=cv_folds, random_state=random_state))

    # stop_words = [gene_placeholder, disease_placeholder]
    # vectorizer = CountVectorizer(stop_words=stop_words)
    vectorizer = CountVectorizer()
    if estimator != 'xgboost':
        if features == 'bow':
            text_clf = Pipeline([('vectorizer', vectorizer),  # ('tfidf', TfidfTransformer()),
                                 ('classifier', clf)])
            text_grid = GridSearchCV(text_clf, param_grid, scoring=scoring, n_jobs=n_jobs, iid=iid, cv=cv_sets)
            _ = text_grid.fit(dataset['sentence_text'], dataset['class'])
        elif features == 'word_vectors':
            text_grid = GridSearchCV(clf, param_grid, scoring=scoring, n_jobs=n_jobs, iid=iid, cv=cv_sets)
            _ = text_grid.fit(matrix_non_missing, dataset['class'])
        else:
            raise ValueError('Unknown features parameter: {}'.format(features))

        results_df = pd.DataFrame(text_grid.cv_results_)
    else:
        out_file = 'tmp_xgboost.tsv'
        _perform_xgboost_cv(dataset, vectorizer, param_grid, scoring, iid, cv_sets, out_file, features,
                            matrix_non_missing)
        xgboost_results = pd.DataFrame.from_csv(out_file, sep='\t', index_col=None)
        param_columns = list(xgboost_results.filter(regex='^param_').columns)
        grouped_results = xgboost_results.groupby(param_columns)
        results_df = grouped_results.apply(_pivot_group).reset_index()

        cv_test_score_columns = list(results_df.filter(regex='_test_score$'))
        results_df['mean_test_score'] = results_df.loc[:, cv_test_score_columns].mean(axis=1, skipna=False)
        results_df['std_test_score'] = results_df.loc[:, cv_test_score_columns].std(axis=1, skipna=False)

        cv_train_score_columns = list(results_df.filter(regex='_train_score$'))
        results_df['mean_train_score'] = results_df.loc[:, cv_train_score_columns].mean(axis=1, skipna=False)
        results_df['std_train_score'] = results_df.loc[:, cv_train_score_columns].std(axis=1, skipna=False)

        results_df['rank_test_score'] = results_df['mean_test_score'].rank(method='first', ascending=False)

    # add CV fold stats to output data frame
    fit_count = len(results_df)
    cv_stats_df = compute_cv_fold_stats(dataset, cv_sets)
    for stats_row in cv_stats_df.itertuples():
        cv_fold = str(stats_row.fold)
        results_df['split_' + cv_fold + '_n_test'] = [stats_row.n_test] * fit_count
        results_df['split_' + cv_fold + '_pos_test'] = [stats_row.pos_test] * fit_count
        results_df['split_' + cv_fold + '_n_train'] = [stats_row.n_train] * fit_count
        results_df['split_' + cv_fold + '_pos_train'] = [stats_row.pos_train] * fit_count

    with gzip.open(results_file, 'wt', encoding='utf-8', errors='raise') as file_out:
        results_df.to_csv(file_out, sep='\t', header=True, index=False)
