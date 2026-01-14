import json
import boto3
import joblib
import pandas as pd
import numpy as np
import shap
from io import BytesIO
from Final_model import x_test

#Lamda

def quick_explain_tree_aws(model, features, x_test_sample):
    """YOUR EXACT SHAP FUNCTION, just returns results instead of printing"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test_sample[0:10])
    
    # Process one customer at a time and append feature score and direction(inc/red) to a list
    customer_1_shap = shap_values[0]
    top_indices = np.argsort(-np.abs(customer_1_shap))[:10]
    
    shap_results = []
    for idx in top_indices:
        feature = features[idx]
        impact = customer_1_shap[idx]
        direction = "INCREASES" if impact > 0 else "REDUCES"
        shap_results.append(f"{feature}: {direction} churn risk by {abs(impact):.2%}")
    
    return shap_results

def get_probability_and_risk_aws(model, x_test_sample, customer_index=0):
    """YOUR EXACT PROBABILITY & RISK CODE"""
    # Get probability for ONE customer
    single_customer_data = x_test_sample[customer_index:customer_index+1]
    prob = model.predict_proba(single_customer_data)[:, 1][0]
    
    # Risk assessment
    if prob > 0.85:
        risk = "Immediate intervention required"
    elif prob > 0.70:
        risk = "Schedule proactive outreach"
    elif prob > 0.50:
        risk = "Monitor closely"
    elif prob > 0.30:
        risk = "Regular monitoring"
    elif prob > 0.15:
        risk = "Below average risks"
    else:
        risk = "Minimal concern"
    
    return prob, risk

#AWS Lambda Handler

def lambda_handler(event, context):
    
    
    s3 = boto3.client('s3')
    model_data = s3.get_object(
        Bucket='churn-model-shreya',
        Key='churn_model.joblib'
    )
    model = joblib.load(BytesIO(model_data['Body'].read()))
    print("Model loaded")
    
    # Get data
    x_test_sample = x_test
    feature_names = event['feature_names']
    customer_index = event.get('customer_index', 0)
    
    print(f"Analyzing {customer_index}")
    
    #run functions 
    shap_explanations = quick_explain_tree_aws(model, feature_names, x_test_sample)
    
    # B) Get probability and risk (your separate code)
    prob, risk = get_probability_and_risk_aws(model, x_test_sample, customer_index)
    
    # 4. Return results in YOUR format
    return {
        "CHURN PROBABILITY": f"{prob:.2%} ({prob:.4f})",
        "RISK ASSESSMENT": risk,
        "TOP 10 SHAP FACTORS": shap_explanations,
        "ANALYSIS FOR": f"Customer {customer_index}",
        "TOTAL CUSTOMERS ANALYZED": len(x_test_sample)
    }