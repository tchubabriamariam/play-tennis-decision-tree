import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

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
# 3. Train black-box classifier
# -----------------------
blackbox = RandomForestClassifier(n_estimators=100, random_state=42)
blackbox.fit(X_train, y_train)
y_pred_blackbox = blackbox.predict(X_test)

# -----------------------
# 4. Generate surrogate training dataset
# -----------------------
# Use black-box predictions on training data as labels
y_train_surrogate = blackbox.predict(X_train)

# -----------------------
# 5. Train Decision Tree surrogate
# -----------------------
# Limit depth for interpretability
surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
surrogate.fit(X_train, y_train_surrogate)

# -----------------------
# 6. Extract rules from surrogate tree
# -----------------------
def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    rules = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            # Left child
            recurse(tree_.children_left[node], path + [(name, "<=", threshold)])
            # Right child
            recurse(tree_.children_right[node], path + [(name, ">", threshold)])
        else:
            # Leaf node: store path and predicted class
            rules.append((path, np.argmax(tree_.value[node][0])))
    recurse(0, [])
    return rules

rules = tree_to_rules(surrogate, X_train.columns)

# Print human-readable rules
print("Extracted rules from surrogate tree:\n")
for path, label in rules:
    rule_str = "IF " + " AND ".join([f"{f} {op} {th:.2f}" for f, op, th in path]) + f" THEN Label={label}"
    print(rule_str)

# -----------------------
# 7. Compute fidelity and accuracy
# -----------------------
def predict_from_rules(rules, sample):
    for path, label in rules:
        match = True
        for f, op, th in path:
            if op == "<=" and not sample[f] <= th:
                match = False
                break
            if op == ">" and not sample[f] > th:
                match = False
                break
        if match:
            return label
    # fallback
    return Counter([label for _, label in rules]).most_common(1)[0][0]

# Apply surrogate rules to test set
X_test_records = X_test.to_dict(orient="records")
y_pred_surrogate = [predict_from_rules(rules, row) for row in X_test_records]

# Fidelity: surrogate vs black-box
fidelity = sum(y_pred_surrogate[i] == y_pred_blackbox[i] for i in range(len(y_test))) / len(y_test)
# Accuracy: surrogate vs true labels
accuracy = sum(y_pred_surrogate[i] == y_test.iloc[i] for i in range(len(y_test))) / len(y_test)

print(f"\nFidelity (vs black-box): {fidelity:.2f}")
print(f"Accuracy (vs true labels): {accuracy:.2f}")
print(f"Number of rules: {len(rules)}")
