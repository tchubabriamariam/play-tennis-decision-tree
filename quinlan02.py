import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# -----------------------
# 1. Load dataset
# -----------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# -----------------------
# 2. Split into train/test
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# -----------------------
# 3. Train black-box
# -----------------------
blackbox = RandomForestClassifier(n_estimators=100, random_state=42)
blackbox.fit(X_train, y_train)
y_pred_blackbox = blackbox.predict(X_test)

# -----------------------
# 4. Discretize features (4 bins for fewer rules)
# -----------------------
def discretize(df, bins=4):
    df_disc = df.copy()
    for col in df.columns:
        df_disc[col] = pd.qcut(df[col], q=bins, labels=False, duplicates='drop')
    return df_disc

X_train_disc = discretize(X_train, bins=4)
X_test_disc = discretize(X_test, bins=4)

# -----------------------
# 5. ID3 with minimum samples to prevent overfitting
# -----------------------
MIN_SAMPLES_LEAF = 5  # stop splitting small subsets

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
        self.children = {}

def build_id3(data, target, features):
    # Stop if subset too small
    if len(data) < MIN_SAMPLES_LEAF:
        return DecisionTreeNode(label=Counter(data[target]).most_common(1)[0][0])
    
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

# Safe prediction with fallback
def predict_id3(node, sample, default=None):
    if node.label is not None:
        return node.label
    val = sample.get(node.feature)
    if val in node.children:
        return predict_id3(node.children[val], sample, default)
    else:
        leaf_labels = [child.label for child in node.children.values() if child.label is not None]
        if leaf_labels:
            return Counter(leaf_labels).most_common(1)[0][0]
        return default

def extract_rules(node, path=""):
    if node.label is not None:
        print(path[:-5] + f" THEN Label={node.label}")
        return
    for val, child in node.children.items():
        extract_rules(child, path + f"IF {node.feature}={val} AND ")

# -----------------------
# 6. Build surrogate on training black-box predictions
# -----------------------
df_surrogate = X_train_disc.copy()
df_surrogate["Label"] = blackbox.predict(X_train)
surrogate_tree = build_id3(df_surrogate, "Label", list(X_train_disc.columns))

most_common_label = Counter(df_surrogate["Label"]).most_common(1)[0][0]

# -----------------------
# 7. Extract human-readable rules
# -----------------------
print("Extracted rules from surrogate tree:")
extract_rules(surrogate_tree)

# -----------------------
# 8. Compute fidelity and accuracy on test set
# -----------------------
X_test_records = X_test_disc.to_dict(orient="records")

correct_fidelity = sum(
    predict_id3(surrogate_tree, row, default=most_common_label) == y_pred_blackbox[i]
    for i,row in enumerate(X_test_records)
)
fidelity = correct_fidelity / len(y_pred_blackbox)

correct_accuracy = sum(
    predict_id3(surrogate_tree, row, default=most_common_label) == y_test.iloc[i]
    for i,row in enumerate(X_test_records)
)
accuracy = correct_accuracy / len(y_pred_blackbox)

# Rule complexity
def count_rules(node):
    if node.label is not None:
        return 1
    return sum(count_rules(child) for child in node.children.values())

num_rules = count_rules(surrogate_tree)

print(f"\nFidelity (vs black-box): {fidelity:.2f}")
print(f"Accuracy (vs true labels): {accuracy:.2f}")
print(f"Number of rules: {num_rules}")
