import pandas as pd

from cocoscore.tools.data_tools import fill_missing_paragraph_document_scores


class TestClass(object):
    input_dict = {
        'class': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 16: 0},
        'distance': {1: -1, 2: 1, 3: 1, 4: -1, 5: 35, 7: -1, 16: 40},
        'entity1': {1: 'DOID:1612',
                    2: 'DOID:1612',
                    3: 'DOID:1612',
                    4: 'DOID:684',
                    5: 'DOID:684',
                    7: 'DOID:684',
                    16: 'DOID:684'},
        'entity2': {1: 'ENSP00000342850',
                    2: 'ENSP00000342850',
                    3: 'ENSP00000342850',
                    4: 'ENSP00000348234',
                    5: 'ENSP00000348234',
                    7: 'ENSP00000348234',
                    16: 'ENSP00000348234'},
        'paragraph': {1: 1, 2: 1, 3: -1, 4: 1, 5: 1, 7: 1, 16: 2},
        'pmid': {1: 8726, 2: 8726, 3: 8726, 4: 13795, 5: 13795, 7: 15997, 16: 17084},
        'predicted': {1: 0.449219,
                      2: 1.0,
                      3: 1.0,
                      4: 0.201172,
                      5: 0.02857142857142857,
                      7: 0.00781252,
                      16: 0.025},
        'sentence': {1: 1, 2: -1, 3: -1, 4: 1, 5: -1, 7: 1, 16: -1},
        'text': {1: 'Identification of MYDISEASETOKEN MYGENETOKEN and its inhibitory role in '
                    'cell-mediated immunity.',
                 2: 'Identification of MYDISEASETOKEN MYGENETOKEN and its inhibitory role in '
                    'cell-mediated immunity.',
                 3: '',
                 4: 'Inhibition by ultraviolet irradiation of the glucocorticoid induction of MYGENETOKEN '
                    'in bromodeoxyuridine-treated H-35 MYDISEASETOKEN.',
                 5: 'Inhibition by ultraviolet irradiation of the glucocorticoid induction of MYGENETOKEN '
                    'in bromodeoxyuridine-treated H-35 MYDISEASETOKEN.',
                 7: 'Effect of concanavalin A on MYGENETOKEN in rat MYDISEASETOKEN tissue culture cells.',
                 16: 'The effect(s) of lack of dietary pyridoxine (PX) on the growth of Morris '
                     'MYDISEASETOKEN no. 7288Ctc was studied. Buffalo strain female rats were fed a diet '
                     'lacking PX. Pair-fed controls were fed the same diet with PX added. Animals were '
                     'inoculated with no. 7288Ctc hepatoma cells at 21 days and were sacrificed 16 days '
                     'later. Host livers and tumors were removed, weights recorded and the activity of '
                     'MYGENETOKEN (TAT; MYGENETOKEN, MYGENETOKEN) was determined in both host liver and '
                     'MYDISEASETOKEN. The average weight of 30 hepatomas grown in pair-fed control rats '
                     'was 11.61 +/- 1.5 g while the average weight of the same number of hepatomas grown '
                     'in animals fed the PX free diet was 4.73 +/- 0.7 g (P less than 0.001). Further TAT '
                     'specific activity levels were 39% and 32% higher in host livers and tumors from '
                     'deficient animals, respectively. The results show that availability of dietary '
                     'pyridoxine stimulates the growth of this MYDISEASETOKEN and, in addition, exercises '
                     'a type of control over the expression of TAT activity.'}
    }

    expected_dict = {
        'class': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
        'distance': {0: 1, 1: 1, 2: -1, 3: -1, 4: 35, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: 40},
        'entity1': {0: 'DOID:1612',
                    1: 'DOID:1612',
                    2: 'DOID:1612',
                    3: 'DOID:684',
                    4: 'DOID:684',
                    5: 'DOID:684',
                    6: 'DOID:684',
                    7: 'DOID:684',
                    8: 'DOID:684',
                    9: 'DOID:684',
                    10: 'DOID:684'},
        'entity2': {0: 'ENSP00000342850',
                    1: 'ENSP00000342850',
                    2: 'ENSP00000342850',
                    3: 'ENSP00000348234',
                    4: 'ENSP00000348234',
                    5: 'ENSP00000348234',
                    6: 'ENSP00000348234',
                    7: 'ENSP00000348234',
                    8: 'ENSP00000348234',
                    9: 'ENSP00000348234',
                    10: 'ENSP00000348234'},
        'paragraph': {0: -1, 1: 1, 2: 1, 3: -1, 4: 1, 5: 1, 6: -1, 7: 1, 8: 1, 9: -1, 10: 2},
        'pmid': {0: 8726,
                 1: 8726,
                 2: 8726,
                 3: 13795,
                 4: 13795,
                 5: 13795,
                 6: 15997,
                 7: 15997,
                 8: 15997,
                 9: 17084,
                 10: 17084},
        'predicted': {0: 1.0,
                      1: 1.0,
                      2: 0.449219,
                      3: 0.0,
                      4: 0.02857142857142857,
                      5: 0.201172,
                      6: 0.0,
                      7: 0.0,
                      8: 0.00781252,
                      9: 0.0,
                      10: 0.025},
        'sentence': {0: -1, 1: -1, 2: 1, 3: -1, 4: -1, 5: 1, 6: -1, 7: -1, 8: 1, 9: -1, 10: -1},
        'text': {0: '',
                 1: 'Identification of MYDISEASETOKEN MYGENETOKEN and its inhibitory role in '
                    'cell-mediated immunity.',
                 2: 'Identification of MYDISEASETOKEN MYGENETOKEN and its inhibitory role in '
                    'cell-mediated immunity.',
                 3: '',
                 4: 'Inhibition by ultraviolet irradiation of the glucocorticoid induction of MYGENETOKEN '
                    'in bromodeoxyuridine-treated H-35 MYDISEASETOKEN.',
                 5: 'Inhibition by ultraviolet irradiation of the glucocorticoid induction of MYGENETOKEN '
                    'in bromodeoxyuridine-treated H-35 MYDISEASETOKEN.',
                 6: '',
                 7: '',
                 8: 'Effect of concanavalin A on MYGENETOKEN in rat MYDISEASETOKEN tissue culture cells.',
                 9: '',
                 10: 'The effect(s) of lack of dietary pyridoxine (PX) on the growth of Morris '
                     'MYDISEASETOKEN no. 7288Ctc was studied. Buffalo strain female rats were fed a diet '
                     'lacking PX. Pair-fed controls were fed the same diet with PX added. Animals were '
                     'inoculated with no. 7288Ctc hepatoma cells at 21 days and were sacrificed 16 days '
                     'later. Host livers and tumors were removed, weights recorded and the activity of '
                     'MYGENETOKEN (TAT; MYGENETOKEN, MYGENETOKEN) was determined in both host liver and '
                     'MYDISEASETOKEN. The average weight of 30 hepatomas grown in pair-fed control rats '
                     'was 11.61 +/- 1.5 g while the average weight of the same number of hepatomas grown '
                     'in animals fed the PX free diet was 4.73 +/- 0.7 g (P less than 0.001). Further TAT '
                     'specific activity levels were 39% and 32% higher in host livers and tumors from '
                     'deficient animals, respectively. The results show that availability of dietary '
                     'pyridoxine stimulates the growth of this MYDISEASETOKEN and, in addition, exercises '
                     'a type of control over the expression of TAT activity.'}
    }

    def test_paragraph_scores_reciprocal(self):
        column_order = ['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'text', 'class',
                        'distance', 'predicted']

        input_df = pd.DataFrame.from_dict(self.input_dict)
        input_df = input_df.loc[:, column_order]

        expected_df = pd.DataFrame.from_dict(self.expected_dict)
        expected_df = expected_df.loc[:, column_order]

        filled_df = fill_missing_paragraph_document_scores(input_df)
        pd.testing.assert_frame_equal(filled_df, expected_df)
