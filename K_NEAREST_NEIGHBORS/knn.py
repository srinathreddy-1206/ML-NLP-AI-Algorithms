#!/usr/bin/env python3
import numpy as np
import operator

def create_data_set():
    group = np.array([[1.0,1.1],[1.0, 1.0],[0,0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file_2_matrix(filename):
    with open(filename, 'r') as fr:
        number_of_lines = len(fr.readlines())
        return_matrix = np.zeros((number_of_lines, 3))
        class_label_vector = []
        index = 0
        fr.seek(0)
        for line in fr.readlines():
            line = line.strip()
            list_from_line = line.split('\t')
            return_matrix[index,:]=list_from_line[:3]
            class_label_vector.append((list_from_line[-1]))
            index+=1
        return return_matrix, class_label_vector
def classify(in_x, data_set, labels, k):
    """
    Pseudo code:
        For every point in our dataset:
            calculate the distance b/w in_x and the current point
            sort the distances in increasing order
            take k items with lowest distances to in_x
            find the majority class among these items
            return the majority class as our prediction for the class of in_x
    """
    data_set_size = data_set.shape[0]
    temp = np.tile(in_x, (data_set_size,1))
    print (temp)
    diff_matrix = np.tile(in_x, (data_set_size, 1)) - data_set
    print (diff_matrix)
    sq_diff_matrix = diff_matrix ** 2
    print (sq_diff_matrix, type(sq_diff_matrix))
    sq_d≡jedi=0, istances = sq_diff_matrix.sum(axis=1)≡ (*value*, sep = ' ', end = '\n', file = sys.stdout) ≡jedi≡
    print (sq_distances)
    distances = sq_distances ** 0.5
    print (distances, type(distances))

    sorted_dist_indices = distances.argsort()
    print (sorted_dist_indices)
    class_count = {}
    for i in range(k):
        vote_ilabel = labels[sorted_dist_indices[i]]
        print (vote_ilabel)
        class_count[vote_ilabel]=class_count.get(vote_ilabel, 0)+1
        print (class_count)

    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse=True)
    print (sorted_class_count)
    return sorted_class_count[0]


if __name__ == "__main__":
    #group, labels = create_data_set()
    group, labels = file_2_matrix('dating_test_set.txt')
    print (group, labels)
    # print(classify([1,1.1], group, labels, 3))
