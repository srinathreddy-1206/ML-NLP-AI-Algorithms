from __future__ import print_function
from math import log
import operator

class DecisionTree(object):
    def __init__(self, data_set = None, ):
        self.data_set = None

    def calculate_shannon_entropy(self,data_set=None):
        if data_set is None: data_set = self.data_set
        num_entries = len(data_set)
        label_counts = {}
        for feat_vec in data_set:
            #print (feat_vec)
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
                reduced_feat_vec.extend(feat_vec[axis+1:])
                return_data_set.append(reduced_feat_vec)
        return return_data_set
    def best_feature_to_split(self, data_set):
        """
            Assumptions Made Here about data_set:
                1. list of lists and all these lists are of equal size
                2. last column in the data or the last item in each instance is the class label of the instance.

        """
        num_features = len(data_set[0])-1
        base_entropy = self.calculate_shannon_entropy(data_set)
        best_info_gain = 0.0; best_feature = -1
        for i in range(num_features):
            feature_list = [example[i] for example in data_set]
            unique_vals = set(feature_list)
            new_entropy = 0.0
            for value in unique_vals:
                sub_data_set =  self.split_data_set(data_set, axis = i, value = value)
                prob = len(sub_data_set)/float(len(data_set))
                new_entropy += prob * self.calculate_shannon_entropy(sub_data_set)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature
    def majority_count(self, class_list):
        """
        takes a list of classes, and returns the class with highest frequency.
        """
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote]=0
            class_count[vote]+=1
        sorted_class_count = sorted(class_count.iteritems(), key = operator.itemgetter(1), reverse = True)
        return sorted_class_count[0][0]
    def create_tree(self, data_set, labels):
        class_list = [example[-1] for example in data_set]
        if class_list.count(class_list[0]) == len(class_list):
            #stop when all classes are equal
            return class_list[0]
        if len(data_set[0]) == 1:
            #when no more features, return majority
            return self.majority_count(class_list)
        best_feature = self.best_feature_to_split(data_set)
        best_feature_label = labels[best_feature]
        my_tree = {best_feature_label:{}}
        del (labels[best_feature])
        #get list of unique vlaues
        feature_values = [example[best_feature] for example in data_set]
        unique_values = set(feature_values)
        for value in unique_values:
            sub_labels = labels[:]
            my_tree[best_feature_label][value]=self.create_tree(self.split_data_set(data_set, best_feature, value), sub_labels)
        return my_tree
if __name__ == "__main__":
    pass
