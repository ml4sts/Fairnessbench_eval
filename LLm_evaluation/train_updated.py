# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder
from aif360.sklearn.metrics import disparate_impact_ratio
import numpy as np
from sklearn.model_selection import GridSearchCV

# Load the training data
train_X = pd.read_csv("train_X.csv")
train_y = pd.read_csv("train_y.csv")
test_X = pd.read_csv("test_X.csv")
test_y = pd.read_csv("test_y.csv")

# Preprocess the data
train_y = train_y.values.ravel()
test_y = test_y.values.ravel()

# Split the data into training and testing sets
#train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# Train the model with hyperparameter tuning to improve disparate impact
param_grid = {
    'class_weight': ['balanced', None],
   'max_iter': [1000, 2000, 3000],
    'C': [0.1, 1, 10]
}

lr = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='f1_macro')
grid_search.fit(train_X, train_y)

# Evaluate the model
best_model = grid_search.best_estimator_
train_pred_y = best_model.predict(train_X)
train_accuracy = accuracy_score(train_y, train_pred_y)
print('Train Accuracy:', train_accuracy)

# Make predictions
test_pred_y = best_model.predict(test_X)

# Evaluate the disparate impact
prot_attr = test_X['sex']
Di = disparate_impact_ratio(y_true=test_y, y_pred=test_pred_y, prot_attr=prot_attr, priv_group=1)
print('Disparate Impact Ratio:', Di)

# Save the predictions
output = pd.DataFrame({
    "Actual_y": test_y,
    "Predicted_y": test_pred_y,
    "sex" : test_X["sex"]
})

output.to_csv("submission.csv", index=False)

# Evaluate the model on the test set
test_accuracy = accuracy_score(test_y, test_pred_y)
print('Test Accuracy:', test_accuracy)