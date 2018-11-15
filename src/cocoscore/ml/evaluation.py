import sklearn.metrics


def precision_recall_reweighted(scores, labels, class_balance_original, class_balance_target):
    """
    Compute precisions, recalls, best F1 score, area under precision-recall curve
    taking class imbalance into account by re-weighting precision.

    See, e.g., paper by Lever et al. [Bioinformatics, btx613, 2017, doi:10.1093/bioinformatics/btx613]
    for motivation behind the re-weighting.

    :param scores: iterable of predicted scores
    :param labels: iterable true class labels: 1 for positives; 0 for negatives
    :param class_balance_original: float positive fraction in original dataset
    :param class_balance_target: float positive fraction in target dataset
    :return: precisions, recalls, best F1 score, area under precision-recall curve
    """
    score_label = sorted(zip(scores, labels), reverse=True)
    positive_count = sum(1 for _, label in score_label if label == 1)
    negative_count = len(score_label) - positive_count

    tp, fp = 0, 0
    best_f_score = -1.0
    precisions = []
    recalls = []
    tp_factor = class_balance_target / class_balance_original
    fp_factor = (1 - class_balance_target) / (1 - class_balance_original)
    for _, label in score_label:
        if label == 1:
            tp += 1
        else:
            fp += 1

        _ = negative_count - fp
        fn = positive_count - tp

        precision, recall, f_score = 0, 0, 0
        if tp + fp != 0:
            precision = tp_factor * tp / float(tp_factor * tp + fp_factor * fp)
        if tp + fn != 0:
            recall = tp / float(tp + fn)
        if precision + recall != 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)

        if f_score > best_f_score:
            best_f_score = f_score

    # add graph points at top left
    precisions.append(1.0)
    recalls.append(0.0)

    rps = sorted(zip(recalls, precisions), reverse=False)
    recalls, precisions = zip(*rps)
    # calculates the area using the trapezoidal rule
    area_under_pr_curve = sklearn.metrics.auc(recalls, precisions)
    return precisions[::-1], recalls[::-1], best_f_score, area_under_pr_curve
