import collections
import itertools

from .entity_mappers import get_serial_to_taxid_name_mapper
from ..tools.file_tools import get_file_handle

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def get_gene_disease_pairs(taxid_name_set, first_taxid=9606, second_taxid=-26):
    first_taxid_names = set()
    second_taxid_names = set()
    for taxid_name in taxid_name_set:
        taxid, name = taxid_name
        if taxid == first_taxid:
            first_taxid_names.add(taxid_name)
        elif taxid == second_taxid:
            second_taxid_names.add(taxid_name)
        else:
            raise ValueError("Encountered unknown taxonomy ID {:d}.".format(taxid))
    return itertools.product(first_taxid_names, second_taxid_names)


def process_current_pmid_score_lines(current_pmid_lines, serial_to_taxid_name):
    return_list = []
    pmid = int(current_pmid_lines[0][0])
    taxid_names = set()
    taxid_name_to_sentences = collections.defaultdict(set)
    taxid_name_to_paragraphs = collections.defaultdict(set)
    for current_line in current_pmid_lines:
        _, paragraph, sentence, _, _, _, taxon, serial = current_line
        taxid_name = serial_to_taxid_name[int(serial)]
        assert int(taxon) == taxid_name[0]
        taxid_names.add(taxid_name)
        taxid_name_to_sentences[taxid_name].add((int(paragraph), int(sentence)))
        taxid_name_to_paragraphs[taxid_name].add(int(paragraph))
    for gene_disease_pair in get_gene_disease_pairs(taxid_names):
        tax_gene, tax_disease = gene_disease_pair
        assert tax_gene[0] == 9606
        assert tax_disease[0] == -26
        if tax_gene == tax_disease:  # this may happen if gene-gene pairs are scored in another project
            continue
        common_sentences = taxid_name_to_sentences[tax_gene] & taxid_name_to_sentences[tax_disease]
        common_paragraphs = taxid_name_to_paragraphs[tax_gene] & taxid_name_to_paragraphs[tax_disease]
        return_list.append([pmid, tax_gene[1], tax_disease[1], common_sentences, common_paragraphs])
    return return_list


def load_matches_file(matches_file_path, entities_file):
    serial_to_taxid_name = get_serial_to_taxid_name_mapper(entities_file)
    matches_file = get_file_handle(matches_file_path, matches_file_path.endswith('.gz'))
    try:
        current_pmid_lines = []
        for line in matches_file:
            # Fields are: pmid, paragraph, sentence, start_match, end_match, matched, taxon, serial
            line_split = line.rstrip().split('\t')
            if len(current_pmid_lines) > 0 and line_split[0] != current_pmid_lines[0][0]:
                yield process_current_pmid_score_lines(current_pmid_lines, serial_to_taxid_name)
                current_pmid_lines = [line_split]
            else:
                current_pmid_lines.append(line_split)
        if len(current_pmid_lines) > 0:
            yield process_current_pmid_score_lines(current_pmid_lines, serial_to_taxid_name)
    finally:
        matches_file.close()


def load_score_file_iterator(score_dict):
    for gene_disease, pmid_paragraph_sentence_dict in score_dict.items():
        gene, disease = gene_disease
        pmid_to_paragraphs = collections.defaultdict(set)
        pmid_to_sentences = collections.defaultdict(set)
        for pmid_paragraph_sentence, _ in pmid_paragraph_sentence_dict.items():
            pmid, paragraph, sentence = pmid_paragraph_sentence
            pmid_to_sentences[pmid].add((paragraph, sentence))
            pmid_to_paragraphs[pmid].add(paragraph)
        for pmid in pmid_to_sentences:
            yield pmid, gene, disease, pmid_to_sentences[pmid], pmid_to_paragraphs[pmid]


def get_max_sentence_score(scores, sentence_co_mentions, pmid, gene, disease):
    sentence_scores = [0.0]
    for paragraph, sentence in sentence_co_mentions:
        key = (pmid, paragraph, sentence)
        if (gene, disease) in scores and key in scores[(gene, disease)]:
            sentence_scores.append(scores[(gene, disease)][(pmid, paragraph, sentence)])
        else:
            # TODO return 0.5 and print message to debug logging if no sentence scores found?
            pass
    return max(sentence_scores)


def get_weighted_counts(matches_file_path, sentence_scores, entities_file, document_weight, paragraph_weight,
                        sentence_weight,
                        ignore_scores=False, silent=False):
    pair_scores = collections.defaultdict(float)
    if matches_file_path is not None:
        matches_iter = load_matches_file(matches_file_path, entities_file)
    else:
        matches_iter = [load_score_file_iterator(sentence_scores)]
    for i, document_matches in enumerate(matches_iter):
        if i > 0 and i % 100000 == 0 and not silent:
            print('Document', i)
        for matches in document_matches:
            pmid, gene, disease, sentence_co_mentions, paragraph_co_mentions = matches
            if isinstance(sentence_scores, dict) and not ignore_scores:
                sentence_score = get_max_sentence_score(sentence_scores, sentence_co_mentions, pmid, gene, disease)
            else:
                if len(sentence_co_mentions) > 0:
                    sentence_score = 1
                else:
                    sentence_score = 0

            if len(paragraph_co_mentions) > 0:
                paragraph_score = 1
            else:
                paragraph_score = 0

            document_score = 1

            pair_score_update = sentence_score * sentence_weight + paragraph_score * paragraph_weight +\
                document_score * document_weight
            pair_scores[(gene, disease)] += pair_score_update
            pair_scores[gene] += pair_score_update
            pair_scores[disease] += pair_score_update
            pair_scores[None] += pair_score_update
    return dict(pair_scores)


def load_score_file(score_file_path):
    compression = score_file_path.endswith('.gz')
    score_file = get_file_handle(score_file_path, compression)
    score_dict = collections.defaultdict(dict)
    try:
        for line in score_file:
            pmid, paragraph, sentence, gene, disease, score = line.rstrip().split('\t')
            score_dict[(gene, disease)][(int(pmid), int(paragraph), int(sentence))] = float(score)
    finally:
        score_file.close()
    return dict(score_dict)


def co_occurrence_score(matches_file_path, score_file_path, entities_file, document_weight=15.0, paragraph_weight=0.0,
                        sentence_weight=1.0, weighting_exponent=0.6, ignore_scores=False, silent=False):
    """
    Computes co-occurrence score for a given matches file and/or sentence score file. See notes from 20170803 for an
    explanation compared to DISEASES scoring scheme (as implemented in co_occurrence_score_diseases).

    :param matches_file_path: matches file as produced by tagger. Used to define co-occurring terms.
    If this is None, co-occurrences are extracted from score_file_path.
    :param score_file_path: sentence score file (tsv formatted) with five columns: pmid, paragraph number, sentence
    number, gene identifier, disease identifier, sentence score
    :param entities_file: entitites file as used by tagger
    :param document_weight: document weight in co-occurrence score
    :param paragraph_weight: paragraph weight in the co-occurrence score
    :param sentence_weight: sentence weight in the co-occurrence score
    :param weighting_exponent: exponent weight in the co-occurrence score
    :param ignore_scores: If True, sentence scores are ignored.
    :param silent: If True, no progress updates are printed
    :return: a dictionary mapping gene, disease pairs to their co-occurrence scores
    """
    if score_file_path is None:
        scores = None
    else:
        scores = load_score_file(score_file_path)
    co_occurrence_scores = {}
    weighted_counts = get_weighted_counts(matches_file_path=matches_file_path, sentence_scores=scores,
                                          entities_file=entities_file,
                                          document_weight=document_weight, paragraph_weight=paragraph_weight,
                                          sentence_weight=sentence_weight,
                                          ignore_scores=ignore_scores, silent=silent)
    norm_factor = weighted_counts[None]
    for key, score in weighted_counts.items():
        if not isinstance(key, tuple):
            continue
        gene, disease = key
        co_occurrence = (score ** weighting_exponent) * \
                        (((score * norm_factor)/(weighted_counts[gene] * weighted_counts[disease])) **
                         (1 - weighting_exponent))
        co_occurrence_scores[key] = co_occurrence
    return co_occurrence_scores


def co_occurrence_score_diseases(matches_file_path, entities_file, document_weight=3.0, paragraph_weight=0.0,
                                 sentence_weight=0.2,
                                 weighting_exponent=0.6,
                                 silent=False):
    return co_occurrence_score(matches_file_path=matches_file_path, score_file_path=None, entities_file=entities_file,
                               document_weight=document_weight,
                               paragraph_weight=paragraph_weight,
                               sentence_weight=sentence_weight, weighting_exponent=weighting_exponent,
                               ignore_scores=True, silent=silent)
