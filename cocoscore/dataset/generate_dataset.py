#!/usr/bin/env python
import argparse
import collections
import gzip
from itertools import product
import logging
import numpy
import os
import pandas

from ..tagger import entity_mappers as em

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'

gene_placeholder = 'MYGENETOKEN'
disease_placeholder = 'MYDISEASETOKEN'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Classifies co-mentions based on their overlap with the given GHR gold standard 
    
    Outputs the following tab-separated columns to stdout:
    'pmid', 'paragraph', 'sentence', 'gene', 'disease', 'sentence_text', 'class' (1 for positive, 0 for negative,
    cases were both disease and gene are not present in GHR gold standard are ignored)
    ''')
    parser.add_argument('entity_file',
                        help='tab-delimited, gzipped file with three columns (serial number, taxonomy ID, entity name) '
                             'as used to specify entities in tagger.')
    parser.add_argument('names_file',
                        help='tab-delimited, gzipped file with two columns (serial number, entity alias) '
                             'as used to specify aliases in tagger.')
    parser.add_argument('preferred_names_file',
                        help='same format as names file but contains only preferred names for certain entities '
                             '(if available)')
    parser.add_argument('ghr_file',
                        help='GHR gene-disease associations from database on cyan. These associations do only include '
                             'those directly listed in GHR and not those inferred from the Disease Ontology. '
                             'File must be gzipped. Ex: data/20170503_GHR/ghr_explicit.tsv.gz')
    parser.add_argument('sentence_file',
                        help='File listing tagger matches and associated sentences as produced '
                             'by dicomclass/medline/extract_sentences.py')
    parser.add_argument('--allow_multiple_pairs', action='store_true',
                        help='Set this if multiple disease-gene pairs are allowed to occur in the same sentence. '
                             'If this is set, a sentence will be classified as 1 if and only if all possible '
                             'disease-gene pairs belong to the positive class. Negative examples are handled '
                             'in the same way.')
    args = parser.parse_args()
    return args.entity_file, args.names_file, args.preferred_names_file, args.ghr_file, args.sentence_file,\
        args.allow_multiple_pairs


def load_ghr_gold_standard_from_disease(diseases_knowledge_file):
    """
    Extract GHR gene-disease associations from DISEASES knowledge channel. 
    
    :param diseases_knowledge_file: tab-delimited, gzipped file with seven columns as downloaded from DISEASES website
    :return: dict: str -> set of str that maps each gene identifier (the ENSP) to the DOIDs it is associated with
    """
    gene_to_diseases = collections.defaultdict(set)
    with gzip.open(diseases_knowledge_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for line in fin:
            gene_id, gene_name, disease_id, disease_name, source, channel, stars = line.rstrip('\n').split('\t')
            if disease_id.startswith('DOID:') and source == 'GHR':
                gene_to_diseases[gene_id].add(disease_id)
    return dict(gene_to_diseases)


def load_ghr_explicit(ghr_file, taxid_alias_to_name):
    """
    Extract GHR gene-disease associations from cyan database. These associations do only include those directly
    listed in GHR and not those inferred from the Disease Ontology.
    
    :param ghr_file: tab-delimited, gzipped file exported from database on cyan. 
    Ex: data/20170503_GHR/ghr_explicit.tsv.gz
    :param taxid_alias_to_name: dict: (int, str) -> str that maps taxonomy ID and alias to final entity name
    :return: dict: str -> set of str that maps each gene identifier (the ENSP) to the DOIDs it is associated with
    """
    gene_to_diseases = collections.defaultdict(set)
    with gzip.open(ghr_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for i, line in enumerate(fin):
            if line.strip() == '':
                continue
            columns = line.rstrip('\n').split('\t')
            type1, id1, type2, id2, source, evidence, score, explicit, url = columns
            assert type1 == '9606' and type2 == '-26'

            if (9606, id1) not in taxid_alias_to_name:
                logging.warning('Gene {} in gold standard file {} could not be mapped.'.format(id1, ghr_file))
                continue
            else:
                id1_mapped = taxid_alias_to_name[(9606, id1)]
            if (-26, id2) not in taxid_alias_to_name:
                logging.warning('Disease {} in gold standard file {} could not be mapped.'.format(id2, ghr_file))
                continue
            else:
                id2_mapped = taxid_alias_to_name[(-26, id2)]

            gene_to_diseases[id1_mapped].add(id2_mapped)
    return dict(gene_to_diseases)


def is_gene(row):
    return row['taxid'] == 9606


def is_gold_standard_gene(row, all_genes):
    return is_gene(row) and (row['name'] in all_genes)


def is_disease(row):
    return row['taxid'] == -26


def is_gold_standard_disease(row, all_diseases):
    return is_disease(row) and (row['name'] in all_diseases)


def extract_data_point(group_df, serial_to_name, taxid_alias_to_name, gene_to_disease, all_genes, all_diseases,
                       allow_multiple_pairs):
    # start with sanity checking the info for the current sentence
    if group_df.ndim == 1:
        logging.warning('Skipping PMID {:d} since only a single match was found.'.format(group_df['pmid']))
        return

    pmid = group_df.loc[:, 'pmid'].iloc[0]
    paragraph = group_df.loc[:, 'paragraph'].iloc[0]
    sentence = group_df.loc[:, 'sentence'].iloc[0]

    unique_sentences = group_df['sentence_text'].unique()
    if len(unique_sentences) != 1:
        logging.warning('Skipping since different sentences corresponding '
                        'to PubMed ID {:d} paragraph{:d} '
                        'sentence {:d} were found: {}{}'.format(pmid, paragraph, sentence, os.linesep,
                                                                (os.linesep + '----' + os.linesep).join(
                                                                    unique_sentences)))
        return

    current_sentence = unique_sentences[0].strip()
    if current_sentence == '':
        logging.warning('Skipping since empty sentence corresponding '
                        'to PubMed ID {:d} paragraph{:d} '
                        'sentence {:d} was found'.format(pmid, paragraph, sentence))
        return

    if '\t' in current_sentence:
        logging.warning('Skipping since tab was found inside sentence, ie. tagger boundaries stretch multiple '
                        'paragraphs corresponding to PubMed ID {:d} paragraph{:d} '
                        'sentence {:d} was found'.format(pmid, paragraph, sentence))
        return

    unique_classes = group_df['taxid'].unique()
    if -26 not in unique_classes:
        logging.warning('Skipping since no disease was found for PubMed ID {:d} paragraph{:d} '
                        'sentence {:d}.'.format(pmid, paragraph, sentence))
        return

    if 9606 not in unique_classes:
        logging.warning('Skipping since no gene was found for PubMed ID {:d} paragraph{:d} '
                        'sentence {:d}.'.format(pmid, paragraph, sentence))
        return

    name_list = []
    for s in group_df['serialno']:
        if s not in serial_to_name:
            logging.info('Could not map serialno {:d} for match in PubMed ID {:d} paragraph{:d} '
                         'sentence {:d}.'.format(s, pmid, paragraph, sentence))
            name_list.append(numpy.NaN)
            continue
        taxid_name = serial_to_name[s]
        if taxid_name not in taxid_alias_to_name:
            logging.info('Could not map alias {} for match in PubMed ID {:d} paragraph{:d} '
                         'sentence {:d}.'.format(taxid_name[1], pmid, paragraph, sentence))
            name_list.append(numpy.NaN)
            continue
        name_list.append(taxid_alias_to_name[taxid_name])

    # find gold standard genes
    group_df['name'] = name_list
    is_gene_vec = group_df.apply(is_gene, axis=1)
    genes = group_df.loc[is_gene_vec, 'name'].unique()
    gene_count = len(genes)
    is_gold_standard_gene_vec = group_df.apply(is_gold_standard_gene, axis=1, args=(all_genes, ))
    gold_standard_genes = group_df.loc[is_gold_standard_gene_vec, 'name'].unique()
    gold_standard_gene_count = len(gold_standard_genes)

    # find gold standard diseases - this requires an additional grouping by tagging position since multiple
    # gold standard diseases may be tagged in one tagging position after propagation up the Disease Ontology tree
    is_disease_vec = group_df.apply(is_disease, axis=1)
    diseases_grouped = group_df.loc[is_disease_vec, :].groupby('first_char')
    gold_standard_diseases = set()
    gold_standard_disease_indices = []
    disease_combinations = []
    for _, disease_group_df in diseases_grouped:
        is_gold_standard_disease_vec = disease_group_df.apply(is_gold_standard_disease, axis=1, args=(all_diseases,))
        if is_gold_standard_disease_vec.sum() == 1:
            gold_standard_diseases.add(str(disease_group_df.loc[is_gold_standard_disease_vec, 'name'].unique()[0]))
            gold_standard_disease_indices.append(disease_group_df.index[is_gold_standard_disease_vec].tolist()[0])
        serialnos = set(disease_group_df.loc[:, 'serialno'])
        if serialnos not in disease_combinations:
            disease_combinations.append(serialnos)

    disease_count = len(disease_combinations)
    gold_standard_diseases = sorted(gold_standard_diseases)
    gold_standard_disease_count = len(gold_standard_diseases)

    curr_gene_df = group_df.loc[is_gold_standard_gene_vec, :]
    curr_disease_df = group_df.loc[gold_standard_disease_indices, :]

    # the if-else-statement below is kept to maintain backwards compatibility with previous runs (the else case below).
    # In theory, the if-case should be able to handle the else-case, too (this assumption untested!).
    if allow_multiple_pairs:
        # check if any genes were tagged that are not part of the gold standard
        if gold_standard_gene_count != gene_count:
            return

        # check if any diseases were tagged that are not part of the gold standard
        if gold_standard_disease_count != disease_count:
            return

        disease_gene_pairs = list(product(gold_standard_diseases, gold_standard_genes))
        disease = '|'.join(gold_standard_diseases)
        gene = '|'.join(gold_standard_genes)
        if len(disease_gene_pairs) == 0:
            logging.warning('Skipping since no gold standard disease-gene co-mention was found for PubMed ID {:d} '
                            'paragraph{:d} sentence {:d}. {} Genes: {} {} Diseases: {}'.format(
                                pmid, paragraph, sentence, os.linesep, gene, os.linesep, disease))
            return
        is_positive = [(d in gene_to_disease[g]) for d, g in disease_gene_pairs]
        is_inconsistent = all(is_positive) != any(is_positive)
        if is_inconsistent:
            logging.info('Skipping since both positive and negative co-mentions were found in PubMed ID {:d} '
                         'paragraph{:d} sentence {:d}.'.format(pmid, paragraph, sentence))
            return
        current_class = int(is_positive[0])
    else:
        if gold_standard_gene_count > 1:
            logging.warning('Skipping since more than one gold standard gene was found for PubMed ID {:d} '
                            'paragraph{:d} '
                            'sentence {:d}.'.format(pmid, paragraph, sentence))
            return

        if gold_standard_disease_count > 1:
            logging.warning('Skipping since more than one gold standard disease was found for PubMed ID {:d} '
                            'paragraph{:d} '
                            'sentence {:d}.'.format(pmid, paragraph, sentence))
            return

        if gold_standard_gene_count + gold_standard_disease_count < 2:
            # gene or disease (or both) not covered by gold standard hence we cannot say whether is as positive/negative
            return

        # decide if P or N data point
        gene = str(curr_gene_df.loc[:, 'name'].iloc[0])
        disease = str(curr_disease_df.loc[:, 'name'].iloc[0])
        if gene in gene_to_disease and disease in gene_to_disease[gene]:
            current_class = 1
        else:
            # note that no checking if disease in gold standard is needed here since this was
            # checked in gold_standard_disease_count > 1 check above
            current_class = 0

    # replace strings that were matched by tagger with fixed placeholders for genes and diseases
    # replacing varying sized tagger matches strings with fixed sized placeholders requires keeping track of the
    # induced offset
    gene_matches = list(zip(curr_gene_df['sentence_first_char'], curr_gene_df['sentence_second_char']))
    disease_matches = list(zip(curr_disease_df['sentence_first_char'], curr_disease_df['sentence_second_char']))
    sorted_matches = sorted(disease_matches + gene_matches)
    assert len(sorted_matches) == len(gene_matches) + len(disease_matches)
    new_sentence = current_sentence
    offset = 0
    for start_end in sorted_matches:
        start, end = start_end
        if start_end in disease_matches:
            new_sentence = new_sentence[:start+offset] + disease_placeholder + new_sentence[end + offset:]
            curr_offset = len(disease_placeholder) - (end - start)
        else:
            new_sentence = new_sentence[:start+offset] + gene_placeholder + new_sentence[end + offset:]
            curr_offset = len(gene_placeholder) - (end - start)
        offset += curr_offset

    print('\t'.join([str(pmid), str(paragraph), str(sentence), gene, disease,
                     new_sentence, str(current_class)]))


def main():
    logging.basicConfig(level=logging.INFO)

    entity_file, names_file, preferred_names_file, ghr_file, sentence_file, allow_multiple_pairs = parse_parameters()

    serial_to_taxid_name = em.get_serial_to_taxid_name_mapper(entity_file)

    taxid_alias_to_name = em.get_taxid_alias_to_name_mapper(names_file, entity_file,
                                                            unique_mappers_only=True,
                                                            preferred_names_file=preferred_names_file)

    gene_to_disease = load_ghr_explicit(ghr_file, taxid_alias_to_name)
    all_genes = set(gene_to_disease.keys())
    all_diseases = {d for disease_set in gene_to_disease.values() for d in disease_set}
    association_count = sum([len(disease_set) for disease_set in gene_to_disease.values()])
    logging.info('Found {:d} gene-disease associations covering {:d} genes and {:d} diseases in '
                 'gold standard file {}.'.format(association_count, len(all_genes), len(all_diseases), ghr_file))

    matches_df = pandas.read_csv(sentence_file, sep='\t', header=None, index_col=None)
    matches_df.columns = ['pmid', 'paragraph', 'sentence', 'first_char',
                          'last_char', 'term', 'taxid', 'serialno',
                          'sentence_first_char', 'sentence_second_char',
                          'sentence_text']

    grouped = matches_df.groupby(['pmid', 'paragraph', 'sentence'])
    for _, g in grouped:
        extract_data_point(g.copy(), serial_to_taxid_name, taxid_alias_to_name, gene_to_disease, all_genes,
                           all_diseases, allow_multiple_pairs)


if __name__ == '__main__':
    main()
