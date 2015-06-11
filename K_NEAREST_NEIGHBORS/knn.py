from __future__ import print_function
import numpy as np
from numpy import array
import operator
from collections import defaultdict
import unittest
import matplotlib
import matplotlib.pyplot as plt

def scatter_plot(data_matrix,labels):
    labels = [int(label) for label in labels]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_matrix[:,1], data_matrix[:,0], 15.0*array(labels), 15.0*array(labels))
    plt.show()

def file_to_matrix(filename, data_set_start=0,data_set_size=3, delimiter='\t', label_position=-1):
    with open(filename, 'r') as f:
        number_of_lines = len(f.readlines())
        return_matrix = np.zeros((number_of_lines,data_set_size))
        class_label_vector = []
        f.seek(0)
        for index, line in enumerate(f.readlines()):
            line=line.strip()
            list_from_line = line.split(delimiter)
            return_matrix[index,:]=list_from_line[data_set_start:data_set_start+data_set_size]
            class_label_vector.append(list_from_line[label_position])
        return return_matrix, class_label_vector


class KNNClassifier(object):
    def __init__(self, data_set, labels, k=3):
        self.data_set = data_set
        self.labels = labels
        self.k=k
        self.normalized_data_set = None
        self.ranges = None
        self.min_vals = None

    def auto_normalize(self):
        #min values of each column
        min_vals = self.data_set.min(0)
        max_vals = self.data_set.max(0)
        ranges = max_vals - min_vals
        norm_data_set = np.zeros(np.shape(self.data_set))
        m = self.data_set.shape[0]
        norm_data_set = self.data_set - np.tile(min_vals,(m,1))
        norm_data_set = norm_data_set/np.tile(ranges, (m,1))
        self.normalized_data_set = norm_data_set
        self.ranges = ranges
        self.min_vals = min_vals

    def classify(self, in_x):
        if self.normalized_data_set is None:
            data_set = self.data_set
        else:
            data_set = self.normalized_data_set
        data_set_size = data_set.shape[0]
        diff_matrix = np.tile(in_x, (data_set_size,1))-data_set
        sq_diff_matrix = diff_matrix ** 2
        sq_distances = sq_diff_matrix.sum(axis=1)
        distances = sq_distances ** 0.5
        sorted_dist_indices = distances.argsort()
        class_count = defaultdict(int)
        for i in range(self.k):
            vote_ilabel =self.labels[sorted_dist_indices[i]]
            class_count[vote_ilabel]+=1
        sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse=True)
        return sorted_class_count[0]

    def classifier_test(self):
        ho_ratio = 0.1
        self.auto_normalize()
        m=self.normalized_data_set.shape[0] #Rows
        num_test_vecs = int(m*ho_ratio) #of Test Vectors
        error_count =  0
        for i in range(num_test_vecs):
            classifier_result = self.classify(self.normalized_data_set[i,:])
            print ("The classifier came back with: %d, the real answer is:: %d"\
                    %(int(classifier_result[0]), int(self.labels[i])))
            if classifier_result[0] != self.labels[i]: error_count+=1
        print("The Total error rate is:%f" %(error_count/float(num_test_vecs)))

class KNNClassifierTests(unittest.TestCase):
    def setUp(self):
        group = np.array([[1., 1.1], [1., 1.],[0,0],[0,0.1]])
        labels = ['A', 'A', 'B', 'B']
        self.classifier = KNNClassifier(group, labels, 3)
    def test_classifier(self):
         self.assertEqual(self.classifier.classify(in_x=[0,0]), ('B', 2))
    def tearDown(self):
        pass

if __name__ =="__main__":
    #unittest.main()
    dating_data_matrix, dating_labels = file_to_matrix('dating_test_set2.txt')
    print (dating_data_matrix[:20], dating_labels[:20])
    scatter_plot(dating_data_matrix,labels = dating_labels)
    classifier=KNNClassifier(dating_data_matrix, dating_labels,20)
    classifier.classifier_test()


