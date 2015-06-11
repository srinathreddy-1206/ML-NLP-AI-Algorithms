from __future__ import print_function
import unittest
from dt import *
def create_sample_date_set():
    data_set = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no'],
        ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

class DecisionTreeTesting(unittest.TestCase):
    def setUp(self):
        self.dt = DecisionTree()
        self.data_set,self.labels = create_sample_date_set()
    def test_entropy(self):
        print(self.dt.calculate_shannon_entropy(self.data_set))
    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

