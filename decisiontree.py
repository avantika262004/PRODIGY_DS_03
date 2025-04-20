import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the smaller dataset
file_path = r"C:\Users\abhin\OneDrive\Desktop\Prodigy infotech\task 3\bank.csv"
data = pd.read_csv(file_path, sep=';')

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data_encoded.drop("y_yes", axis=1)
y = data_encoded["y_yes"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(class_weight='balanced',random_state=42, max_depth=3)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=["No", "Yes"], 
          filled=True, 
          fontsize=9, 
          max_depth=5) 

plt.title("Decision Tree View")
plt.show()
plt.savefig("decision_tree.png")