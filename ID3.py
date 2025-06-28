import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        label_counts = np.bincount(labels)
        probabilities = label_counts / len(labels)
        entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        attribute_values = [data[attribute] for data in dataset]
        unique_values = set(attribute_values)
        average_entropy = 0.0
        for value in unique_values:
            subset = [dataset[i] for i in range(len(dataset)) if dataset[i][attribute] == value]
            subset_labels = [labels[i] for i in range(len(labels)) if dataset[i][attribute] == value]
            subset_entropy = self.calculate_entropy__(subset, subset_labels)
            average_entropy += (len(subset) / len(dataset)) * subset_entropy
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        total_entropy = self.calculate_entropy__(dataset, labels)
        average_entropy = self.calculate_average_entropy__(dataset, labels, attribute)
        information_gain = total_entropy - average_entropy
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        attribute_values = [data[attribute] for data in dataset]
        unique_values = set(attribute_values)
        intrinsic_info = 0.0
        for value in unique_values:
            subset = [dataset[i] for i in range(len(dataset)) if dataset[i][attribute] == value]
            intrinsic_info -= (len(subset) / len(dataset)) * np.log2(len(subset) / len(dataset))
        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        information_gain = self.calculate_information_gain__(dataset, labels, attribute)
        intrinsic_info = self.calculate_intrinsic_information__(dataset, labels, attribute)
        if intrinsic_info == 0:
            return 0
        gain_ratio = information_gain / intrinsic_info
        return gain_ratio

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        # If all instances have the same label, create a leaf node
        if len(set(labels)) == 1:
            return TreeLeafNode(dataset, labels[0])
        
        # If all features are used, create a leaf node with the majority label
        if len(used_attributes) == len(self.features):
            majority_label = max(set(labels), key=labels.count)
            return TreeLeafNode(dataset, majority_label)
        
        # Determine the best attribute to split
        best_attribute = None
        best_score = -1
        for attribute in range(len(self.features)):
            if attribute in used_attributes:
                continue
            
            # Calculate the score based on the chosen criterion
            if self.criterion == "information gain":
                score = self.calculate_information_gain__(dataset, labels, attribute)
            else:
                score = self.calculate_gain_ratio__(dataset, labels, attribute)
            
            # Update the best attribute if a better score is found
            if score > best_score:
                best_score = score
                best_attribute = attribute
        
        # Log the chosen attribute and criterion
        print(f"Splitting on attribute '{self.features[best_attribute]}' using '{self.criterion}' with score {best_score:.4f}")
        
        # Mark the attribute as used and create a node
        used_attributes.append(best_attribute)
        node = TreeNode(best_attribute)
        
        # Recursively create subtrees for each unique value of the best attribute
        attribute_values = [data[best_attribute] for data in dataset]
        unique_values = set(attribute_values)
        for value in unique_values:
            subset = [dataset[i] for i in range(len(dataset)) if dataset[i][best_attribute] == value]
            subset_labels = [labels[i] for i in range(len(labels)) if dataset[i][best_attribute] == value]
            node.subtrees[value] = self.ID3__(subset, subset_labels, used_attributes.copy())
        
        return node


    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        node = self.root
        while isinstance(node, TreeNode):
            attribute_value = x[node.attribute]
            if attribute_value in node.subtrees:
                node = node.subtrees[attribute_value]
            else:
                break
        if isinstance(node, TreeLeafNode):
            return node.labels
        return max(set(node.labels), key=node.labels.count)

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")