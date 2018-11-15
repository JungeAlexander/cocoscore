#!/usr/bin/env python

import cocoscore.tagger.entity_mappers as ent


class TestClass(object):
    names_file = 'tests/tagger/names.tsv'
    entity_file = 'tests/tagger/entities.tsv'
    names_file_compressed = 'tests/tagger/names.tsv.gz'
    entities_file_compressed = 'tests/tagger/entities.tsv.gz'

    def test_basic_serial(self):
        obtained = ent.get_serial_to_taxid_name_mapper(self.entity_file, compressed=False)
        assert {1000: (9606, '--D'), 1: (-26, 'A'), 10: (-26, ' B-  b ')} == obtained

    def test_basic_serial_reverse(self):
        obtained = ent.get_taxid_name_to_serial_mapper(self.entity_file, compressed=False)
        assert {(9606, '--D'): 1000, (-26, 'A'): 1, (-26, ' B-  b '): 10} == obtained

    def test_basic_entity(self):
        obtained = ent.get_taxid_alias_to_name_mapper(self.names_file, self.entity_file, unique_mappers_only=True,
                                                      compressed=False)
        assert {(9606, '--D'): '--D', (-26, 'A'): 'A', (-26, ' B-  b '): ' B-  b '} == obtained

    def test_allow_non_unique_mappers(self):
        obtained = ent.get_taxid_alias_to_name_mapper(self.names_file, self.entity_file, unique_mappers_only=False,
                                                      compressed=False, lowercase_strip=False)
        assert {(9606, '--D'): {'--D'}, (-26, 'A'): {'A'}, (-26, ' B-  b '): {' B-  b '},
                (-26, 'a'): {'A', ' B-  b '}} == obtained

    def test_lowercase_strip_entity(self):
        obtained = ent.get_taxid_alias_to_name_mapper(self.names_file, self.entity_file, unique_mappers_only=True,
                                                      compressed=False, lowercase_strip=True)
        assert {(9606, 'd'): '--D', (-26, 'bb'): ' B-  b '} == obtained

    def test_gzip(self):
        obtained = ent.get_taxid_alias_to_name_mapper(self.names_file_compressed, self.entities_file_compressed,
                                                      unique_mappers_only=True,
                                                      compressed=True)
        assert {(9606, '--D'): '--D', (-26, 'A'): 'A', (-26, ' B-  b '): ' B-  b '} == obtained
