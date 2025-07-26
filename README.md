# WisconsinBreastCancer
This project applies supervised machine learning to classify breast cancer tumors as Malignant (M) or Benign (B) using the Wisconsin Breast Cancer dataset. To handle class imbalance in the dataset, the ADASYN (Adaptive Synthetic Sampling) technique was employed. The classification model used is a Random Forest Classifier, fine-tuned via RandomizedSearchCV for optimal performance.

📊 Dataset
Source: Wisconsin Diagnostic Breast Cancer (WDBC) dataset

Features: 30 real-valued input features derived from digitized images of a fine needle aspirate (FNA) of a breast mass.

Target: diagnosis (1 = Malignant, 0 = Benign)

Initial class distribution:

Benign (0): 357

Malignant (1): 212

⚙️ Preprocessing Steps
Dropped non-informative ID column

Encoded diagnosis labels (M → 1, B → 0)

Applied Min-Max normalization to scale features to [0, 1]

Handled missing values (if any) by imputing with zeros

🧪 Train-Test Split
Stratified split to preserve class proportions

80% training, 20% testing

🧬 Imbalanced Data Handling
Used ADASYN to generate synthetic samples for the minority class (Malignant)

Resulting in a balanced training set with both classes having similar counts

🌲 Classifier: Random Forest
Hyperparameters optimized using RandomizedSearchCV over:

n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap

5-fold cross-validation

✅ Model Evaluation
Achieved high classification performance on the unseen test set:

yaml
Copy
Edit
Test Accuracy: 98.25%
Confusion Matrix:

lua
Copy
Edit
[[71  1]
 [ 1 41]]
Classification Report:

markdown
Copy
Edit
              precision    recall  f1-score   support

           0       0.99      0.99      0.99        72
           1       0.98      0.98      0.98        42

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
📌 Technologies Used
Python

pandas, numpy

scikit-learn

imbalanced-learn (ADASYN)

matplotlib

🚀 How to Run
bash
Copy
Edit
pip install -r requirements.txt
python BreastCancer.py
