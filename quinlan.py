import numpy as np
import pandas as pd
from collections import Counter

import time

# Example dataset: Play Tennis (classic Quinlan example)
data = pd.DataFrame([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
], columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

X = data.drop('PlayTennis', axis=1)
y = data['PlayTennis']

def entropy(target_col):
    counts = Counter(target_col)
    total = len(target_col)
    return -sum((count/total) * np.log2(count/total) for count in counts.values())

def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = sum(
        (len(data[data[feature]==v])/len(data)) * entropy(data[data[feature]==v][target])
        for v in values
    )
    return total_entropy - weighted_entropy

class DecisionTreeNode:
    def __init__(self, feature=None, value=None, left=None, right=None, *, label=None):
        self.feature = feature      # feature to split on
        self.value = value          # value for categorical split (if needed)
        self.left = left
        self.right = right
        self.label = label          # label if leaf node

def build_tree(data, target, features):
    labels = data[target].unique()
    # Stop conditions
    if len(labels) == 1:
        return DecisionTreeNode(label=labels[0])
    if len(features) == 0:
        # majority vote
        return DecisionTreeNode(label=Counter(data[target]).most_common(1)[0][0])
    
    # Select feature with max information gain
    gains = [info_gain(data, f, target) for f in features]
    best_feature = features[np.argmax(gains)]
    
    tree = DecisionTreeNode(feature=best_feature)
    
    for val in data[best_feature].unique():
        subset = data[data[best_feature]==val]
        if subset.empty:
            leaf = DecisionTreeNode(label=Counter(data[target]).most_common(1)[0][0])
            setattr(tree, val, leaf)
        else:
            subtree = build_tree(subset, target, [f for f in features if f != best_feature])
            setattr(tree, val, subtree)
    
    return tree

def extract_rules(node, path=""):
    if node.label is not None:
        print(path + " THEN PlayTennis = " + node.label)
        return
    for val in node.__dict__.keys():
        if val in ['feature', 'value', 'left', 'right', 'label']:
            continue
        child = getattr(node, val)
        extract_rules(child, path + f"IF {node.feature} = {val} AND ")

features = list(X.columns)
tree = build_tree(data, 'PlayTennis', features)
extract_rules(tree)

# Measure speed
start = time.time()
tree = build_tree(data, 'PlayTennis', features)
training_time = time.time() - start
print("Training time:", training_time)

# Measure accuracy 
def predict(tree, sample):
    node = tree
    while node.label is None:
        feature_val = sample[node.feature]
        node = getattr(node, feature_val)
    return node.label

correct = 0
for idx, row in data.iterrows():
    if predict(tree, row) == row['PlayTennis']:
        correct += 1
accuracy = correct / len(data)
print("Accuracy:", accuracy)

# Fidelity = 100% because rules come directly from the tree
fidelity = 1.0
print("Fidelity:", fidelity)

test_samples = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'},
    {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong'},
    {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak'}
]

for s in test_samples:
    print(s, "=>", predict(tree, s))


#                 Outlook
#           /        |         \
#        Sunny    Overcast     Rain
#         /           |          \
#    Humidity        Yes         Wind
#     /    \                     /   \
#  High   Normal              Weak   Strong
#   No       Yes               Yes      No
