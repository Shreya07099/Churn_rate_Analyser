#Importing necessary libraries
from cleaning import df
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
from sklearn.metrics import make_scorer, average_precision_score
#Data Preprocessing- replace all white spaces with underscores
df['Contract'].replace("-","_",regex=True,inplace=True)
'''print(df['InternetService'].unique())
print(df["Contract"].unique())
print(df["PaymentMethod"].unique())'''
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.replace(' ', '_', regex=False)
#print(df['SeniorCitizen'])

#Formatting data- spliting into dependent and independent variable type
x=df.drop("Churn",axis=1).copy()#Dont lose original table
#print(x.head)
y=df["Churn"].copy()
#print(y.head)
#print(x.info)


#OneHotEncoding
x_encoded=pd.get_dummies(x,columns=['gender','SeniorCitizen',
                                    'Partner','Dependents',
                                    'PhoneService','MultipleLines',
                                    'InternetService','OnlineSecurity',
                                    'OnlineBackup','DeviceProtection',
                                    'TechSupport','StreamingTV',
                                    'StreamingMovies','Contract',
                                    'PaperlessBilling','PaymentMethod'])
#print(x_encoded.head)

#Splitting the training and testing data using stratify 
#print(sum(y)/len(y))
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,random_state=42,stratify=y)
#print(sum(y_train)/len(y_train))
#print(sum(y_test)/len(y_test))
clf_xgb=xgb.XGBClassifier(objective='binary:logistic',seed=42)
clf_xgb.fit(x_train,y_train,verbose=True,early_stopping_rounds=10,eval_metric='aucpr',eval_set=[(x_test,y_test)])
print(clf_xgb.get_booster().best_iteration)
print(clf_xgb.get_booster().attributes())

#Plotting Confusion_Matrix
y_proba = clf_xgb.predict_proba(x_test)[:,1]
y_pred = (y_proba > 0.5).astype(int)

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Did not leave","Left"],
    cmap="Blues", values_format="d"
)
plt.show()
#Tells us the metrics: inference there is a class imbalance
#METRICS
#1169----122
#245----222

#optimization of the model
param_grid_round1 = {
    'max_depth': [3,4,5],
    'learning_rate': [0.1, 0.01, 0.05],  # general ranges for all
    'gamma': [0,0.25,1],
    'reg_lambda': [0,1,10],   
    'scale_pos_weight': [1,3,5]
}

# Example GridSearchCV setup:


model = XGBClassifier(
    n_estimators=100,  # or higher if needed
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

'''grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid_round1,
    scoring='roc_auc',  # AUC as recommended
    cv=5,
    n_jobs=-1,
    verbose=2
)'''

'''grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("\nAll Results:")
results_df = pd.DataFrame(grid_search.cv_results_)'''
#print(results_df[['params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score'))
#Results: Best parameters: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 1}


'''param_grid_refined = {
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.2],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [0, 0.1, 1],
    'scale_pos_weight': [1, 1.5, 2]  # small adjustments
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid_refined,
    scoring=aucpr_scorer,  # AUC as recommended
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)'''
param_grid_final = {
    'max_depth': [2],
    'learning_rate': [0.05],
    'gamma':  [0.1],
    'reg_lambda': [0.5,4,8,10],
    'scale_pos_weight': [2, 2.5, 2.85,3]  # scale_pos_weight = negative_samples / positive_samples==aprox 2.85
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid_final,
    scoring='roc_auc',  # AUC as recommended
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)
