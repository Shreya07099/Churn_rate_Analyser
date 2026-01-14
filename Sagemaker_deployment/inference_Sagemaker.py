import json           
import joblib         
import pandas as pd   
import numpy as np    
import os             
import sys            
import shap            
model = None           # already trained
feature_names = None   # json has it
x_test_sample = None   # is in the form of csv


def model_fn(model_dir):

    global model, feature_names, x_test_sample
    
  

    model_file = "churn_model.joblib" 
    model_path = r"C:\Users\shre0\vs_code sign\Churn_rate_Analyser\Sagemaker_deployment\churn_model.joblib"
    
    print(f"Loading model: {model_file}")
    model = joblib.load(model_path)
    print(f"Model loaded: {type(model)}")
    
    features_file = "feature_names.json" 
    features_path = r"C:\Users\shre0\vs_code sign\Churn_rate_Analyser\Sagemaker_deployment\feature_names.json"
    
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    print(f"Features loaded: {len(feature_names)} features")
    
    
    test_file = "x_test_data.csv"  
    test_path = r"C:\Users\shre0\vs_code sign\Churn_rate_Analyser\Sagemaker_deployment\x_test_data.csv"
    
    # Load CSV - NO HEADER (just numbers)
    x_test_sample = pd.read_csv(test_path).values


    print(f"Loaded: {model_file}, {features_file}, {test_file}")
    print(f"Ready to predict for {x_test_sample.shape[0]} customers")
    
    return model

def quick_explain_tree_aws(customer_index=0):
    global model, x_test_sample, feature_names    
    try:

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test_sample[0:10])
        customer_1_shap = shap_values[customer_index]
        top_indices = np.argsort(-np.abs(customer_1_shap))[:10]
        shap_results = []
        for idx in top_indices:  # Loop through top 10 features
            feature = feature_names[idx]  # Get feature name
            impact = customer_1_shap[idx]  # Get impact value
            direction = "INCREASES" if impact > 0 else "REDUCES"  # Positive/negative
            shap_results.append(f"{feature}: {direction} churn risk by {abs(impact):.2%}")
        
        return shap_results  # Return list of explanations
    except Exception as e:
        return [f"SHAP Error: {str(e)}"]  
def get_probability_and_risk_aws(customer_index=0):
    global model, x_test_sample
    single_customer_data = x_test_sample[customer_index:customer_index+1]
    prob = model.predict_proba(single_customer_data)[:, 1][0]
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
    
    return prob, risk  # Return probability number and risk category

def input_fn(request_body, request_content_type):

    try:
        customer_index = int(request_body.strip())#convert to int
    except:
        raise ValueError("Please send JUST a number")
    if customer_index < 0:#validate range
        customer_index = 0
    
    if customer_index >= len(x_test_sample):
        customer_index = len(x_test_sample) - 1
    return {'customer_index': customer_index}
def predict_fn(input_data, model):
    global x_test_sample, feature_names
    customer_index = input_data
    if isinstance(input_data, dict):
        customer_index = input_data.get('customer_index', 0)
    else:
        customer_index = int(input_data)
    try:

        # 2. Get probability and risk (YOUR FUNCTION)
        prob, risk = get_probability_and_risk_aws(customer_index)
        
        # 3. Get SHAP explanations for THIS customer (modified)
        # Pass customer_index instead of feature_names
        shap_explanations = quick_explain_tree_aws(customer_index)
        
        # 4. Compile results
        result = {
            "CUSTOMER_INDEX": customer_index,
            "CHURN_PROBABILITY": f"{prob:.1%}",
            "PROBABILITY_RAW": float(prob),
            "RISK_LEVEL": risk,
            "RECOMMENDATION": "HIGH PRIORITY" if prob > 0.5 else "LOW PRIORITY",
            "TOP_SHAP_FACTORS": shap_explanations,
                    }
        return result
        

        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "CUSTOMER_INDEX": customer_index,
            "ERROR": str(e),
            "STATUS": "FAILED"
        }
def output_fn(prediction, response_content_type):
        print(f"📤 Formatting output as {response_content_type}")
        
        if response_content_type == 'application/json':#dict to  json format
            return json.dumps(prediction, indent=2)  # indent=2 makes it pretty
        else:
            raise ValueError(f"Unsupported output type: {response_content_type}")
    

