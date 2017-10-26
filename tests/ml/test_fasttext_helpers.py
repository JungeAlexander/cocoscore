import pandas
import unittest

import cocoscore.ml.fasttext_helpers as fth


class CVTest(unittest.TestCase):
    train_path = 'ft_simple_test.txt'
    ft_path = '/home/lib/fastText'
    model_path = 'testmodel'
    test_path = 'ft_simple_test.txt'
    probability_path = 'ft_simple_prob.txt'

    def test_train_call_parameters(self):
        train_call, compress_call = fth.get_fasttext_train_calls(self.train_path, {'-aaa': 1.0}, self.ft_path,
                                                                 self.model_path, thread=5)
        expected_train_call = self.ft_path + ' supervised -input ' + self.train_path + ' -output ' + self.model_path + \
            ' -aaa 1.0 -thread 5 '
        self.assertEqual(train_call, expected_train_call)
        expected_compress_call = self.ft_path + ' quantize -input ' + self.model_path + ' -output ' + self.model_path
        self.assertEqual(compress_call, expected_compress_call)

    def test_test_call_parameters(self):
        predict_call = fth.get_fasttext_test_calls(self.test_path, self.ft_path, self.model_path, self.probability_path)
        expected_predict_call = self.ft_path + ' predict-prob ' + self.model_path + ' ' + self.test_path + ' ' + \
            str(2) + ' | gzip > ' + self.probability_path
        self.assertEqual(predict_call, expected_predict_call)


if __name__ == '__main__':
    unittest.main()
