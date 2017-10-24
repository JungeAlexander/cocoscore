from sklearn.metrics.ranking import _binary_clf_curve


def fp_tp_curve(true_classes, scores, pos_label=1):
    """
    True positive and false positive counts for different classification thresholds.
    This is just a wrapper for sklearn.metrics.ranking._binary_clf_curve so far.

    :param true_classes: true binary labels
    :param scores: predicted scores
    :param pos_label: label considered as positive, everything else is considered negative
    :return: increasing false positive counts, increasing true positive counts, decreasing thresholds
    """
    fps, tps, thresholds = _binary_clf_curve(true_classes, scores, pos_label=pos_label, sample_weight=None)
    return fps, tps, thresholds
