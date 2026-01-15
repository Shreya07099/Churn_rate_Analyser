import json           
import joblib         
import pandas as pd   
import numpy as np    
import os             
import sys            
import shap   
import boto3         
model = None           # already trained
feature_names = None   # json has it
x_test_sample = None   # is in the form of csv
sns_client=None
SNS_TOPIC_ARN=None

def model_fn(model_dir):

    global model, feature_names, x_test_sample,sns_client,SNS_TOPIC_ARN
    
    model_path = os.path.join(model_dir, "churn_model.joblib")
    # Build path to model file inside the package
    model = joblib.load(model_path)
    model_file = "churn_model.joblib"
    features_file = "feature_names.json" 
    features_path = os.path.join(model_dir, "feature_names.json")
    
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    print(f"Features loaded: {len(feature_names)} features")
    
    
    test_file = "x_test_data.csv"  
    test_path = os.path.join(model_dir, "x_test_data.csv")
    
    # Load CSV - NO HEADER (just numbers)
    x_test_sample = pd.read_csv(test_path).values
    print(f"Loaded: {model_file}, {features_file}, {test_file}")
    try:
        SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', 
                                      'arn:aws:sns:us-east-1:YOUR-ACCOUNT-ID:Churn-Alerts')
        sns_client = boto3.client('sns', region_name='us-east-1')
        print(f"SNS initialized")
    except Exception as e:
        print(f"SNS failed: {str(e)}")
        sns_client = None

    return model

def send_churn_alert(customer_index, probability, risk_level, shap_factors):
    global sns_client, SNS_TOPIC_ARN
    
    if sns_client is None or probability <= 0.5:
        return False
    
    try:
        subject = f"CHURN ALERT: Customer #{customer_index} - {probability:.1%} Risk"
        message = f"""
CUSTOMER CHURN RISK ALERT
Customer Index: #{customer_index}
Churn Probability: {probability:.1%}
Risk : {risk_level}

 TOP RISK FACTORS:
{chr(10).join([f"  • {factor}" for factor in shap_factors[:3]])}
"""
        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        print(f"Alert sent: {response['MessageId']}")
        return True
    except Exception as e:
        print(f"Alert failed: {str(e)}")
        return False


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
        alert_sent = send_churn_alert(customer_index, prob, risk, shap_explanations)
        
        # 4. Compile results
        result = {
            "CUSTOMER_INDEX": customer_index,
            "CHURN_PROBABILITY": f"{prob:.1%}",
            "PROBABILITY_RAW": float(prob),
            "RISK_LEVEL": risk,
            "RECOMMENDATION": "HIGH PRIORITY" if prob > 0.5 else "LOW PRIORITY",
            "EMAIL_ALERT_SENT": alert_sent,
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
    

