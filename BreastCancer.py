import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

# Load data
data = pd.read_csv("D:\\Datasets\\Wisconsin\\data.csv")
print(data.head())

# Drop ID and map diagnosis
data.drop(['id'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
y = data['diagnosis'].values
x_data = data.drop(['diagnosis'], axis=1)

# Check class distribution
class_counts = data['diagnosis'].value_counts()
print("Class distribution (actual counts):\n", class_counts)

# ✅ Safe normalization
x = x_data.apply(lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0)

# ✅ Fill any remaining NaNs (just in case)
x = x.fillna(0)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# ✅ Confirm no NaNs in x_train
print("NaNs in x_train:", x_train.isnull().sum().sum())

# Apply ADASYN
adasyn = ADASYN(random_state=42)
x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train, y_train)

# Resampled class distribution
print("Class distribution after ADASYN:\n", pd.Series(y_train_resampled).value_counts())

# CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit model
random_search.fit(x_train_resampled, y_train_resampled)

# Best parameters
print("Best parameters found:", random_search.best_params_)

# Predict on test set
y_pred = random_search.predict(x_test)

# Evaluate performance
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))