#!/usr/bin/env python
import collections
import gzip
import re

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'

_whitespace_hyphen_pattern = re.compile(r'\s+|-')


def reverse_dict(input_dict):
    return_dict = collections.defaultdict(set)
    for k, v in input_dict.items():
        if isinstance(v, set):
            for val in v:
                return_dict[val].add(k)
        else:
            return_dict[v].add(k)
    return dict(return_dict)


def lowercase_strip_str(string):
    return re.sub(_whitespace_hyphen_pattern, '', string).lower()


def get_file_handle(file_path, compression):
    if compression:
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='replace')
    else:
        return open(file_path, 'rt', encoding='utf-8', errors='replace')


def get_serial_to_taxid_name_mapper(entity_file, taxids=(9606, -26), compressed=True):
    """
    Returns a mapper for serial numbers to their final names.

    :param entity_file: tab-delimited file with three columns (serial number, taxonomy ID, entity name).
    :param taxids: list of integers, the taxonomy IDs to use.
    :param compressed: indicates whether files are gzipped or not
    :return: dict: int -> (int, str) that maps serial numbers to the corresponding taxonomy ID and entity name.
    """
    serial_to_taxid_name = {}
    fin = get_file_handle(entity_file, compressed)
    try:
        for line in fin:
            serialno, taxid, name = line.rstrip('\n').split('\t')
            serialno = int(serialno)
            taxid = int(taxid)
            if taxid not in taxids:
                continue
            if serialno in serial_to_taxid_name:
                raise ValueError('Serial number {:d} is duplicated in entity file {}.'.format(serialno, entity_file))
            serial_to_taxid_name[serialno] = (taxid, name)
    finally:
        fin.close()
    return serial_to_taxid_name


def get_taxid_name_to_serial_mapper(entity_file, taxids=(9606, -26), compressed=True):
    """
    Returns a mapper for final names to their serials numbers.

    :param entity_file: tab-delimited file with three columns (serial number, taxonomy ID, entity name).
    :param taxids: list of integers, the taxonomy IDs to use.
    :param compressed: indicates whether files are gzipped or not
    :return: dict: int -> (int, str) that maps taxonomy ID and entity name to the corresponding serial numbers.
    """
    serial_to_taxid_name = get_serial_to_taxid_name_mapper(entity_file=entity_file, taxids=taxids,
                                                           compressed=compressed)
    taxid_name_to_serial = {}
    for serial, taxid_name in serial_to_taxid_name.items():
        if taxid_name in taxid_name_to_serial:
            raise ValueError('Name {} (taxon {}) is duplicated in entity file {}.'.format(taxid_name[1], taxid_name[0],
                                                                                          entity_file))
        taxid_name_to_serial[taxid_name] = serial
    return taxid_name_to_serial


def _get_tax_id_preferred_name_to_name(preferred_names_file, serial_to_taxid_name, compressed):
    fin = get_file_handle(preferred_names_file, compressed)
    tax_id_preferred_name_to_name = {}
    try:
        for line in fin:
            serial_no, pref_name = line.rstrip('\n').split('\t')
            serial_no = int(serial_no)
            if serial_no not in serial_to_taxid_name:
                # comes from another taxon
                continue
            taxid, name = serial_to_taxid_name[serial_no]
            key = (taxid, pref_name)
            # FIXME The sanity check below is triggered for yeast (tax ID 4932), thus remove it for now
            # if key in tax_id_preferred_name_to_name and tax_id_preferred_name_to_name[key] != name:
            #     raise ValueError('Preferred name {} refers to multiple entities. Preferred names file: {} '.format(
            #         pref_name, preferred_names_file))
            tax_id_preferred_name_to_name[key] = name
    finally:
        fin.close()
    return tax_id_preferred_name_to_name


def get_taxid_alias_to_name_mapper(names_file, entity_file, unique_mappers_only=True, taxids=(9606, -26),
                                   lowercase_strip=False, preferred_names_file=None, compressed=True):
    """
    Returns a mapper for aliases to their final names.

    :param names_file: tab-delimited file with two columns (serial number, entity alias). Note that serial
    numbers must not be unique (if the file lists several alternative names for the same entity).
    :param entity_file: tab-delimited file with three columns (serial number, taxonomy ID, entity name).
    :param taxids: list of integers, the taxonomy IDs to use.
    :param unique_mappers_only: indicates whether aliases that map to multiple names should be included in the returned
    hash.
    :param lowercase_strip: indicates whether aliases should be converted to lowercase, stripped of whitespace and
    hyphens in the final mapping.
    :param preferred_names_file: path to file with preferred names which has the same format as names_file. Whenever an
    alias is declared a preferred name in this file, the alias will not be mapped to other names in the same taxon.
    :param compressed: indicates whether files are gzipped or not
    :return: If unique_mappers_only is True, dict: (int, str) -> str that maps taxonomy ID and alias to final entity
    name.
    Otherwise, dict: (int, str) -> set of str that maps taxonomy ID and alias to the final entity name(s).
    """
    if unique_mappers_only:
        taxid_alias_to_name = {}
    else:
        taxid_alias_to_name = collections.defaultdict(set)
    remove_keys = set()
    serial_to_taxid_name = get_serial_to_taxid_name_mapper(entity_file, taxids=taxids, compressed=compressed)

    fin = get_file_handle(names_file, compressed)
    try:
        for line in fin:
            serial_no, alias = line.rstrip('\n').split('\t')
            serial_no = int(serial_no)
            if serial_no not in serial_to_taxid_name:
                # comes from another taxon
                continue
            taxid, name = serial_to_taxid_name[serial_no]
            if lowercase_strip:
                key = (taxid, lowercase_strip_str(alias))
            else:
                key = (taxid, alias)
            if key in taxid_alias_to_name and taxid_alias_to_name[key] != name and unique_mappers_only:
                remove_keys.add(key)
            if unique_mappers_only:
                taxid_alias_to_name[key] = name
            else:
                taxid_alias_to_name[key].add(name)
    finally:
        fin.close()

    # Ensure that all names in entity_file map to itself
    for taxid, name in serial_to_taxid_name.values():
        if lowercase_strip:
            key = (taxid, lowercase_strip_str(name))
        else:
            key = (taxid, name)
        if unique_mappers_only:
            if key in taxid_alias_to_name and taxid_alias_to_name[key] != name:
                remove_keys.add(key)
            taxid_alias_to_name[key] = name
        else:
            taxid_alias_to_name[key].add(name)

    for k in remove_keys:
        del taxid_alias_to_name[k]

    # If a preferred names file was specified, make sure that those aliases map to their preferred names and
    # nothing else
    if preferred_names_file is not None:
        tax_id_preferred_name_to_name = _get_tax_id_preferred_name_to_name(preferred_names_file, serial_to_taxid_name,
                                                                           compressed=compressed)
        for tax_alias, name in tax_id_preferred_name_to_name.items():
            taxid, alias = tax_alias
            if lowercase_strip:
                key = (taxid, lowercase_strip_str(alias))
            else:
                key = (taxid, alias)
            if unique_mappers_only:
                taxid_alias_to_name[key] = name
            else:
                taxid_alias_to_name[key] = {name}

    # Do not return a defaultdict instance as this may lead to unexpected side effects
    if unique_mappers_only:
        return taxid_alias_to_name
    else:
        return dict(taxid_alias_to_name)
