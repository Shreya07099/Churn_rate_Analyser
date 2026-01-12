from main import x_train, y_train
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Create simple model
model = XGBClassifier(
    max_depth=2,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)

# Train on your data
model.fit(x_train, y_train)

# Make predictions
y_pred_proba = model.predict_proba(x_train)[:, 1]  # Probabilities for class 1

# Calculate AUC
auc = roc_auc_score(y_train, y_pred_proba)
print(f"AUC on training data: {auc}")