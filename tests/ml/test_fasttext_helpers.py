import os
import unittest

import cocoscore.ml.fasttext_helpers as fth


class CVTest(unittest.TestCase):
    train_path = 'tests/ml/ft_simple_test.txt'
    ft_path = 'fasttext'
    model_path = 'fastText_test_model'
    test_path = train_path
    probability_path = 'ft_simple_prob.txt.gz'

    def test_train_call_parameters(self):
        train_call, compress_call = fth.get_fasttext_train_calls(self.train_path, {'-aaa': 1.0}, self.ft_path,
                                                                 self.model_path, thread=5)
        expected_train_call = self.ft_path + ' supervised -input ' + self.train_path + ' -output ' + self.model_path + \
            ' -aaa 1.0 -thread 5'
        self.assertEqual(' '.join(train_call), expected_train_call)
        expected_compress_call = self.ft_path + ' quantize -input ' + self.model_path + ' -output ' + self.model_path
        self.assertEqual(' '.join(compress_call), expected_compress_call)

    def test_test_call_parameters(self):
        predict_call = fth.get_fasttext_test_calls(self.test_path, self.ft_path, self.model_path)
        expected_predict_call = self.ft_path + ' predict-prob ' + self.model_path + ' ' + self.test_path + ' ' + \
            str(2)
        self.assertEqual(' '.join(predict_call), expected_predict_call)

    def test_fit(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        expected_model_path = self.model_path + '.bin'
        self.assertEqual(model_path, expected_model_path)
        self.assertTrue(os.path.isfile(model_path))
        os.remove(model_path)

    def test_fit_compressed(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000, '-wordNgrams': 2}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=True)
        expected_model_path = self.model_path + '.ftz'
        self.assertEqual(model_path, expected_model_path)
        self.assertTrue(os.path.isfile(model_path))
        os.remove(model_path)

    def test_predict(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        fth.fasttext_predict(model_path, self.test_path, self.ft_path, self.probability_path)
        self.assertTrue(os.path.isfile(self.probability_path))
        os.remove(model_path)
        os.remove(self.probability_path)

    def test_fasttext_class_probabilities(self):
        model_path = fth.fasttext_fit(self.train_path, {'-bucket': 1000}, self.ft_path,
                                      model_path=self.model_path, thread=1,
                                      compress_model=False)
        fth.fasttext_predict(model_path, self.test_path, self.ft_path, self.probability_path)
        probabilities = fth.load_fasttext_class_probabilities(self.probability_path)
        self.assertEqual(len(probabilities), 40)
        self.assertTrue(all([x > 0.75 for x in probabilities[:20]]))
        self.assertTrue(all([x < 0.25 for x in probabilities[20:]]))
        os.remove(model_path)
        os.remove(self.probability_path)


if __name__ == '__main__':
    unittest.main()
