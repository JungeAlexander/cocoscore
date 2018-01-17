import collections
import itertools

from .entity_mappers import get_serial_to_taxid_name_mapper
from ..tools.file_tools import get_file_handle

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def get_entity_pairs(type_name_set, first_type, second_type):
    first_type_names = set()
    second_type_names = set()
    for type_name in type_name_set:
        mytype, name = type_name
        if mytype == first_type:
            first_type_names.add(type_name)
        elif mytype == second_type:
            second_type_names.add(type_name)
        else:
            raise ValueError("Encountered unknown type {:d}.".format(mytype))
    if first_type != second_type:
        return itertools.product(first_type_names, second_type_names)
    else:
        return itertools.combinations(first_type_names, 2)


def process_current_pmid_score_lines(current_pmid_lines, serial_to_type_entity, first_type, second_type):
    return_list = []
    pmid = int(current_pmid_lines[0][0])
    type_entities = set()
    type_entity_to_sentences = collections.defaultdict(set)
    type_entity_to_paragraphs = collections.defaultdict(set)
    for current_line in current_pmid_lines:
        _, paragraph, sentence, _, _, _, mytype, serial = current_line
        type_entity = serial_to_type_entity[int(serial)]
        assert int(mytype) == type_entity[0]
        type_entities.add(type_entity)
        type_entity_to_sentences[type_entity].add((int(paragraph), int(sentence)))
        type_entity_to_paragraphs[type_entity].add(int(paragraph))
    for entity_pair in get_entity_pairs(type_entities, first_type, second_type):
        first_type_entity, second_type_entity = entity_pair
        assert first_type_entity[0] == first_type
        assert second_type_entity[0] == second_type
        common_sentences = type_entity_to_sentences[first_type_entity] & type_entity_to_sentences[second_type_entity]
        common_paragraphs = type_entity_to_paragraphs[first_type_entity] & type_entity_to_paragraphs[second_type_entity]
        entity_key = sorted((first_type_entity[1], second_type_entity[1]))
        return_list.append([pmid, *entity_key, common_sentences, common_paragraphs])
    return return_list


def load_matches_file(matches_file_path, entities_file, first_type, second_type):
    serial_to_type_name = get_serial_to_taxid_name_mapper(entities_file, taxids=(first_type, second_type))
    matches_file = get_file_handle(matches_file_path, matches_file_path.endswith('.gz'))
    try:
        current_pmid_lines = []
        for line in matches_file:
            # Fields are: pmid, paragraph, sentence, start_match, end_match, matched, type, serial
            line_split = line.rstrip().split('\t')
            if len(current_pmid_lines) > 0 and line_split[0] != current_pmid_lines[0][0]:
                yield process_current_pmid_score_lines(current_pmid_lines, serial_to_type_name, first_type,
                                                       second_type)
                current_pmid_lines = [line_split]
            else:
                current_pmid_lines.append(line_split)
        if len(current_pmid_lines) > 0:
            yield process_current_pmid_score_lines(current_pmid_lines, serial_to_type_name, first_type, second_type)
    finally:
        matches_file.close()


def load_sentence_score_iterator(score_dict):
    for entity_pair, pmid_paragraph_sentence_dict in score_dict.items():
        entity_1, entity_2 = entity_pair
        pmid_to_paragraphs = collections.defaultdict(set)
        pmid_to_sentences = collections.defaultdict(set)
        for pmid_paragraph_sentence, _ in pmid_paragraph_sentence_dict.items():
            pmid, paragraph, sentence = pmid_paragraph_sentence
            pmid_to_sentences[pmid].add((paragraph, sentence))
            pmid_to_paragraphs[pmid].add(paragraph)
        for pmid in pmid_to_sentences:
            yield pmid, entity_1, entity_2, pmid_to_sentences[pmid], pmid_to_paragraphs[pmid]


def load_paragraph_score_iterator(score_dict):
    for entity_pair, pmid_paragraph_dict in score_dict.items():
        entity_1, entity_2 = entity_pair
        pmid_to_paragraphs = collections.defaultdict(set)
        for pmid_paragraph, _ in pmid_paragraph_dict.items():
            pmid, paragraph = pmid_paragraph
            pmid_to_paragraphs[pmid].add(paragraph)
        for pmid in pmid_to_paragraphs:
            yield pmid, entity_1, entity_2, {}, pmid_to_paragraphs[pmid]


def load_document_score_iterator(score_dict):
    for entity_pair, pmid_dict in score_dict.items():
        entity_1, entity_2 = entity_pair
        for pmid in pmid_dict.keys():
            yield pmid, entity_1, entity_2, {}, {}


def get_max_sentence_score(scores, sentence_co_mentions, pmid, entity_1, entity_2):
    sentence_scores = [0.0]
    for paragraph, sentence in sentence_co_mentions:
        key = (pmid, paragraph, sentence)
        if (entity_1, entity_2) in scores and key in scores[(entity_1, entity_2)]:
            sentence_scores.append(scores[(entity_1, entity_2)][key])
        else:
            # TODO return 0.5 and print message to debug logging if no sentence scores found?
            pass
    return max(sentence_scores)


def get_max_paragraph_score(scores, paragraph_co_mentions, pmid, entity_1, entity_2):
    paragraph_scores = [0.0]
    for paragraph in paragraph_co_mentions:
        key = (pmid, paragraph)
        if (entity_1, entity_2) in scores and key in scores[(entity_1, entity_2)]:
            paragraph_scores.append(scores[(entity_1, entity_2)][key])
        else:
            # TODO return 0.5 and print message to debug logging if no paragraph scores found?
            pass
    return max(paragraph_scores)


def get_document_score(scores, pmid, entity_1, entity_2):
    key = pmid
    if (entity_1, entity_2) in scores and key in scores[(entity_1, entity_2)]:
        return scores[(entity_1, entity_2)][key]
    else:
        # TODO return 0.5 and print message to debug logging if no document scores found? 
        pass


def get_weighted_counts(matches_file_path, sentence_scores, paragraph_scores, document_scores,
                        entities_file, first_type, second_type,
                        document_weight, paragraph_weight, sentence_weight,
                        ignore_scores=False, silent=False):
    pair_scores = collections.defaultdict(float)
    if matches_file_path is not None:
        matches_iter = load_matches_file(matches_file_path, entities_file, first_type, second_type)
    else:
        if sentence_scores is not None:
            my_iterator = load_sentence_score_iterator(sentence_scores)
        elif paragraph_scores is not None:
            my_iterator = load_paragraph_score_iterator(paragraph_scores)
        else:
            my_iterator = load_document_score_iterator(document_scores)
        matches_iter = [my_iterator]
    for i, document_matches in enumerate(matches_iter):
        if i > 0 and i % 100000 == 0 and not silent:
            print('Document', i)
        for matches in document_matches:
            pmid, entity_1, entity_2, sentence_co_mentions, paragraph_co_mentions = matches
            if isinstance(sentence_scores, dict) and not ignore_scores:
                sentence_score = get_max_sentence_score(sentence_scores, sentence_co_mentions, pmid, entity_1, entity_2)
            else:
                if len(sentence_co_mentions) > 0:
                    sentence_score = 1
                else:
                    sentence_score = 0

            if isinstance(paragraph_scores, dict) and not ignore_scores:
                paragraph_score = get_max_paragraph_score(paragraph_scores, paragraph_co_mentions, pmid, entity_1,
                                                          entity_2)
            else:
                if len(paragraph_co_mentions) > 0:
                    paragraph_score = 1
                else:
                    paragraph_score = 0

            if isinstance(document_scores, dict) and not ignore_scores:
                document_score = get_document_score(document_scores, pmid, entity_1, entity_2)
            else:
                document_score = 1

            pair_score_update = sentence_score * sentence_weight + paragraph_score * paragraph_weight +\
                document_score * document_weight
            pair_scores[(entity_1, entity_2)] += pair_score_update
            pair_scores[entity_1] += pair_score_update
            pair_scores[entity_2] += pair_score_update
            pair_scores[None] += pair_score_update
    return dict(pair_scores)


def load_sentence_score_file(score_file_path):
    compression = score_file_path.endswith('.gz')
    score_file = get_file_handle(score_file_path, compression)
    score_dict = collections.defaultdict(dict)
    try:
        for line in score_file:
            pmid, paragraph, sentence, entity_1, entity_2, score = line.rstrip().split('\t')
            entity_key = tuple(sorted((entity_1, entity_2)))
            score_dict[entity_key][(int(pmid), int(paragraph), int(sentence))] = float(score)
    finally:
        score_file.close()
    return dict(score_dict)


def load_paragraph_score_file(score_file_path):
    compression = score_file_path.endswith('.gz')
    score_file = get_file_handle(score_file_path, compression)
    score_dict = collections.defaultdict(dict)
    try:
        for line in score_file:
            pmid, paragraph, entity_1, entity_2, score = line.rstrip().split('\t')
            entity_key = tuple(sorted((entity_1, entity_2)))
            score_dict[entity_key][(int(pmid), int(paragraph))] = float(score)
    finally:
        score_file.close()
    return dict(score_dict)


def load_document_score_file(score_file_path):
    compression = score_file_path.endswith('.gz')
    score_file = get_file_handle(score_file_path, compression)
    score_dict = collections.defaultdict(dict)
    try:
        for line in score_file:
            pmid, entity_1, entity_2, score = line.rstrip().split('\t')
            entity_key = tuple(sorted((entity_1, entity_2)))
            score_dict[entity_key][int(pmid)] = float(score)
    finally:
        score_file.close()
    return dict(score_dict)


def co_occurrence_score(matches_file_path, sentence_score_file_path, paragraph_score_file_path,
                        document_score_file_path,
                        entities_file, first_type, second_type,
                        document_weight=15.0, paragraph_weight=0.0,
                        sentence_weight=1.0, weighting_exponent=0.6, ignore_scores=False, silent=False):
    """
    Computes co-occurrence score for a given matches file and/or sentence score file. See notes from 20170803 for an
    explanation compared to DISEASES scoring scheme (as implemented in co_occurrence_score_diseases).

    :param matches_file_path: matches file as produced by tagger. Used to define co-occurring terms.
    If this is None, co-occurrences are extracted from score_file_path.
    :param sentence_score_file_path: sentence score file (tsv formatted) with five columns: pmid, paragraph number,
    sentence number, first entity, second entity, sentence score
    :param paragraph_score_file_path: paragraph score file (tsv formatted) with five columns: pmid, paragraph number,
    first entity, second entity, sentence score
    :param document_score_file_path: sentence score file (tsv formatted) with five columns: pmid, first entity,
    second entity, sentence score
    :param entities_file: entities file as used by tagger
    :param first_type: int, type of the first entity class to be scored
    :param second_type: int, type of the second entity class to be scored
    :param document_weight: document weight in co-occurrence score
    :param paragraph_weight: paragraph weight in the co-occurrence score
    :param sentence_weight: sentence weight in the co-occurrence score
    :param weighting_exponent: exponent weight in the co-occurrence score
    :param ignore_scores: If True, sentence scores are ignored.
    :param silent: If True, no progress updates are printed
    :return: a dictionary mapping entity pairs to their co-occurrence scores
    """
    if sentence_score_file_path is None:
        sentence_scores = None
    else:
        sentence_scores = load_sentence_score_file(sentence_score_file_path)
    if paragraph_score_file_path is None:
        paragraph_scores = None
    else:
        paragraph_scores = load_paragraph_score_file(paragraph_score_file_path)
    if document_score_file_path is None:
        document_scores = None
    else:
        document_scores = load_document_score_file(document_score_file_path)
    co_occurrence_scores = {}
    weighted_counts = get_weighted_counts(matches_file_path=matches_file_path, sentence_scores=sentence_scores,
                                          paragraph_scores=paragraph_scores, document_scores=document_scores,
                                          entities_file=entities_file, first_type=first_type, second_type=second_type,
                                          document_weight=document_weight, paragraph_weight=paragraph_weight,
                                          sentence_weight=sentence_weight,
                                          ignore_scores=ignore_scores, silent=silent)
    norm_factor = weighted_counts[None]
    for key, score in weighted_counts.items():
        if not isinstance(key, tuple):
            continue
        entity_1, entity_2 = key
        co_occurrence = (score ** weighting_exponent) * \
                        (((score * norm_factor)/(weighted_counts[entity_1] * weighted_counts[entity_2])) **
                         (1 - weighting_exponent))
        co_occurrence_scores[key] = co_occurrence
    return co_occurrence_scores


def co_occurrence_score_diseases(matches_file_path, entities_file, document_weight=3.0, paragraph_weight=0.0,
                                 sentence_weight=0.2,
                                 weighting_exponent=0.6,
                                 silent=False):
    return co_occurrence_score(matches_file_path=matches_file_path, sentence_score_file_path=None,
                               paragraph_score_file_path=None, document_score_file_path=None,
                               entities_file=entities_file,
                               first_type=9606, second_type=-26,
                               document_weight=document_weight,
                               paragraph_weight=paragraph_weight,
                               sentence_weight=sentence_weight, weighting_exponent=weighting_exponent,
                               ignore_scores=True, silent=silent)


def co_occurrence_score_string(matches_file_path, entities_file, entity_type, document_weight=1.0, paragraph_weight=2.0,
                               sentence_weight=0.2, weighting_exponent=0.6, silent=False):
    return co_occurrence_score(matches_file_path=matches_file_path, sentence_score_file_path=None,
                               paragraph_score_file_path=None, document_score_file_path=None,
                               entities_file=entities_file,
                               first_type=entity_type, second_type=entity_type,
                               document_weight=document_weight, paragraph_weight=paragraph_weight,
                               sentence_weight=sentence_weight, weighting_exponent=weighting_exponent,
                               ignore_scores=True, silent=silent)
