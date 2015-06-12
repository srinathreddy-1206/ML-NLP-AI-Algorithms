from __future__ import print_function
import unittest
from dt import *
def create_sample_data_set():
    data_set = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no'],
        ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

class DecisionTreeTest(unittest.TestCase):
    def setUp(self):
        self.dt = DecisionTree()
        self.data_set,self.labels = create_sample_data_set()

    def test_entropy(self):
        self.assertEqual(round(self.dt.calculate_shannon_entropy(self.data_set), 2), 0.97 )

    def test_split_data(self):
        test_data = [
                        [1, 1, 'yes'],
                        [1, 1, 'yes'],
                        [1, 0, 'no'],
                        [0, 1, 'no'],
                        [0, 1, 'no'],
                    ]
        self.assertEqual(self.dt.split_data_set(test_data, axis=0, value = 1), [[1, 'yes'],[1, 'yes'],[0, 'no']])
        self.assertEqual(self.dt.split_data_set(test_data, axis=0, value = 0), [[1, 'no'],[1, 'no']])

    def test_best_feature_to_split(self):
        data, labels = create_sample_data_set()
        self.assertEqual (self.dt.best_feature_to_split(data),0)
    def test_create_tree(self):
        data, labels = create_sample_data_set()
        tree = self.dt.create_tree(data, labels)
        print (tree)
    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

