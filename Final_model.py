from cleaning import df
from main import x_train, y_train, x_test,y_test
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import os

clf_xgb = xgb.XGBClassifier(
    seed=42,
    objective='binary:logistic',
    gamma=0.1,
    learning_rate=0.05,  
    max_depth=2,
    reg_lambda=0.5,
    scale_pos_weight=2,
    subsample=0.9,
    colsample_bytree=0.5,
    n_estimators=1000,  # Use large number with early stopping
    eval_metric='aucpr',
    use_label_encoder=False
)

# Train with early stopping
print("Training XGBoost model with early stopping...")
clf_xgb.fit(
    x_train,
    y_train,
    verbose=True,  #
    early_stopping_rounds=10,
    eval_set=[(x_test, y_test)]  # Using test set for validation (consider using validation set instead)
)

print(f"\nBest iteration: {clf_xgb.best_iteration}")

#Plotting Confusion_Matrix
y_proba = clf_xgb.predict_proba(x_test)[:,1]
y_pred = (y_proba > 0.5).astype(int)

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Did not leave","Left"],
    cmap="Blues", values_format="d"
)
plt.show()

#Final Inference: Significant Improvement in detecting customers that are actually going to leave(rest documentation on README)
#Saving Model
import joblib
joblib.dump(clf_xgb, 'churn_model.joblib')
print("Model saved")
print("at:", os.path.abspath('churn_model.joblib'))
import json
feature_names = x_train.columns.tolist()  #
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("Feature names saved")
print("at:", os.path.abspath('feature_names.json'))