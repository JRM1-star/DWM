from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)
# Create Decision Tree model
clf = DecisionTreeClassifier(criterion=&quot;gini&quot;, max_depth=3,
random_state=42)
# Train model
clf.fit(X_train, y_train)
# Test model
accuracy = clf.score(X_test, y_test)
print(&quot;Decision Tree Accuracy:&quot;, accuracy)
# Visualize Decision Tree

plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names,
class_names=iris.target_names, filled=True)
plt.show()
