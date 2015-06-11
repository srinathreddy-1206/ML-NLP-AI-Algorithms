from __future__ import print_function
from math import log
class DecisionTree(object):
    def __init__(self, data_set = None, ):
        self.data_set = None

    def calculate_shannon_entropy(self,data_set=None):
        if data_set is None: data_set = self.data_set
        num_entries = len(data_set)
        label_counts = {}
        for feat_vec in data_set:
            print (feat_vec)
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label]=0
            label_counts[current_label]+=1
        shannon_entropy = 0.0
        for key in label_counts.keys():
            prob = float(label_counts[key])/num_entries
            shannon_entropy -= prob * log(prob, 2)
        return shannon_entropy
    def split_data_set(self, data_set, axis, value):
        return_data_set = []
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                reduced_feat_vec = feat_vec[:axis]
                reduced_feat_vec.extend(feat_vec[axis+1])
                return_data_set.append(reduced_feat_vec)
        return return_data_set

if __name__ == "__main__":
    pass
