# Decision Tree Classification with Python and Scikit-Learn

## Project Overview
This project implements a Decision Tree Classifier to predict car safety. Two models are built—one using the Gini Index and another using the Entropy criterion. The implementation is done using Python and Scikit-Learn, and the Car Evaluation Data Set from the UCI Machine Learning Repository is used.

## Table of Contents
1. Introduction to Decision Tree Algorithm
2. Classification and Regression Trees (CART)
3. Decision Tree Algorithm Intuition
4. Attribute Selection Measures
   - Information Gain
   - Gini Index
5. Problem Statement
6. Dataset Description
7. Implementation Steps
   - Importing Libraries
   - Loading Dataset
   - Exploratory Data Analysis
   - Splitting Data
   - Feature Engineering
   - Model Training (Gini & Entropy)
   - Model Evaluation
   - Results and Conclusion

## 1. Introduction to Decision Tree Algorithm
A Decision Tree is a popular machine learning algorithm that uses a tree-like structure for decision-making. It belongs to the class of supervised learning algorithms and is used for both classification and regression tasks.

## 2. Classification and Regression Trees (CART)
The modern term for Decision Trees is CART, introduced by Leo Breiman. It is the foundation for other models like Random Forest and Boosted Decision Trees.

## 3. Decision Tree Algorithm Intuition
- Each dataset attribute forms a node.
- The most significant attribute is placed at the root node.
- Decisions are made by traversing the tree from the root to a leaf node.

## 4. Attribute Selection Measures
### Information Gain
- Measures impurity reduction using entropy.
- The attribute with the highest information gain is selected for splitting.

### Gini Index
- Measures the probability of misclassification.
- The attribute with the lowest Gini Index is chosen.

## 5. Problem Statement
The task is to predict car safety levels using the Decision Tree Classifier. The Car Evaluation Data Set is used, obtained from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation).

## 6. Dataset Description
- The dataset consists of 1,728 instances and 7 categorical features:
  - `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`, `class`
- The target variable (`class`) represents the safety level of the car (`unacc`, `acc`, `good`, `vgood`).

## 7. Implementation Steps
### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

### Loading Dataset
```python
df = pd.read_csv('car.data', header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
```

### Exploratory Data Analysis
```python
df.info()
df.head()
df['class'].value_counts()
```

### Splitting Data
```python
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Feature Engineering
- Convert categorical variables into numerical format using encoding.

### Model Training
#### Decision Tree (Gini Index)
```python
gini_clf = DecisionTreeClassifier(criterion='gini', random_state=42)
gini_clf.fit(X_train, y_train)
```
#### Decision Tree (Entropy)
```python
entropy_clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
entropy_clf.fit(X_train, y_train)
```

### Model Evaluation
```python
y_pred_gini = gini_clf.predict(X_test)
y_pred_entropy = entropy_clf.predict(X_test)

print(confusion_matrix(y_test, y_pred_gini))
print(classification_report(y_test, y_pred_gini))
```

## 8. Results and Conclusion
- The Decision Tree Classifier performed well on the dataset.
- The entropy-based model had slightly better accuracy than the Gini-based model.
- Feature importance analysis can further improve the model.

## 9. References
- UCI Machine Learning Repository: [Car Evaluation Dataset](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
- Scikit-Learn Documentation: [Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html)

## 10. Contact
- For any inquiries or support, reach out via:
- Email: hamzaaslam276jb@gmail.com
- GitHub: [Hamza Aslam](https://github.com/hamzaaslam276)
