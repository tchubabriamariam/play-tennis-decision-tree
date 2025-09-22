import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------
# 1. Dataset
# -----------------------
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

# -----------------------
# 2. Encode categorical features for black-box
# -----------------------
encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
X_enc = pd.DataFrame({col: encoders[col].transform(X[col]) for col in X.columns})
y_enc = LabelEncoder().fit_transform(y)

# -----------------------
# 3. Train black-box (Random Forest)
# -----------------------
blackbox = RandomForestClassifier(n_estimators=50, random_state=42)
blackbox.fit(X_enc, y_enc)
y_blackbox = blackbox.predict(X_enc)  # predictions on the same dataset

# -----------------------
# 4. Quinlan ID3 surrogate tree implementation
# -----------------------
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
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label
        self.children = {}  # value -> subtree

def build_id3(data, target, features):
    labels = data[target].unique()
    if len(labels) == 1:
        return DecisionTreeNode(label=labels[0])
    if not features:
        return DecisionTreeNode(label=Counter(data[target]).most_common(1)[0][0])
    gains = [info_gain(data, f, target) for f in features]
    best_feature = features[np.argmax(gains)]
    node = DecisionTreeNode(feature=best_feature)
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        if subset.empty:
            node.children[val] = DecisionTreeNode(label=Counter(data[target]).most_common(1)[0][0])
        else:
            node.children[val] = build_id3(subset, target, [f for f in features if f != best_feature])
    return node

def extract_rules(node, path=""):
    if node.label is not None:
        print(path[:-5] + f" THEN Label={node.label}")
        return
    for val, child in node.children.items():
        extract_rules(child, path + f"IF {node.feature}={val} AND ")

def predict_id3(node, sample):
    if node.label is not None:
        return node.label
    val = sample[node.feature]
    if val in node.children:
        return predict_id3(node.children[val], sample)
    else:
        # fallback to majority label of children
        return Counter([child.label for child in node.children.values() if child.label]).most_common(1)[0][0]

# -----------------------
# 5. Build surrogate tree on black-box predictions
# -----------------------
df_surrogate = X.copy()
df_surrogate["Label"] = y_blackbox
surrogate_tree = build_id3(df_surrogate, "Label", list(X.columns))

# -----------------------
# 6. Extract rules
# -----------------------
print("Extracted Rules (Quinlan ID3 Surrogate):")
extract_rules(surrogate_tree)

# -----------------------
# 7. Compute fidelity and accuracy
# -----------------------
# Fidelity: agreement with black-box
correct_fidelity = sum(predict_id3(surrogate_tree, row)==y_blackbox[i] 
                       for i,row in enumerate(X.to_dict(orient="records")))
fidelity = correct_fidelity / len(y_blackbox)

# Accuracy: agreement with true labels
correct_accuracy = sum(predict_id3(surrogate_tree, row)==y_enc[i] 
                       for i,row in enumerate(X.to_dict(orient="records")))
accuracy = correct_accuracy / len(y_blackbox)

# Rule complexity
def count_rules(node):
    if node.label is not None:
        return 1
    return sum(count_rules(child) for child in node.children.values())

num_rules = count_rules(surrogate_tree)

print(f"\nFidelity (vs black-box): {fidelity:.2f}")
print(f"Accuracy (vs true labels): {accuracy:.2f}")
print(f"Number of rules: {num_rules}")
