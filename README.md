# Churn_rate_Analyser
Author-Shreya Nair, VIT Vellore
# Customer Retention & Churn Prediction System

## Project Overview

Developed a comprehensive customer retention solution using XGBoost to predict and prevent churn among high-value customers. The system integrates multiple data sources including transaction history, behavioral patterns, and customer support interactions to provide accurate churn predictions with actionable insights.

## Technical Implementation

### Machine Learning Architecture
- **Model Development**: Implemented XGBoost algorithm optimized for imbalanced churn prediction data
- **Feature Engineering**: Created predictive features from integrated transaction, behavioral, and support data streams
- **Explainable AI**: Incorporated SHAP analysis to provide transparent model interpretations and churn driver identification

### System Infrastructure
- **Data Pipeline**: Built ETL processes for real-time data integration from multiple sources
- **Cloud Deployment**: Implemented serverless AWS architecture using Lambda functions for model inference
- **Notification System**: Configured Amazon SNS for automated email re-engagement campaigns
- **Event Processing**: Designed event-driven workflow triggering retention actions based on prediction scores

## Key Features

### Multi-Source Data Integration
- Unified customer view combining financial transactions, engagement metrics, and support interactions
- Real-time data processing for up-to-date churn risk assessment


### Explainable Predictions
- SHAP-based interpretations providing clear churn driver analysis
- Actionable insights for customer success teams
- Transparent model decisions supporting business trust

### Automated Intervention System
- Real-time churn risk scoring and alerting
- Personalized email campaign triggering


## Technical Stack
- **Machine Learning**: XGBoost, SHAP, Scikit-learn, Pandas, NumPy
- **Cloud Services**: AWS Lambda, Amazon SNS, S3, EventBridge
- **Development Tools**: Python, Git, SQL
- **Visualization**: Matplotlib, Seaborn for model diagnostics

## Files includes
data_cleaning.py - Data preprocessing module for cleaning and merging customer transaction, behavioral, and support data.

explain_shap_model.py - SHAP implementation for model interpretability and feature importance visualization.

Final_model.py - final ready XGBoost model after hyperparameter optimization with inference functions.

main.py - Main pipeline orchestrating data preprocessing, one hotencoding, and various iterations of hyperparameter optimizations.

data/ folder - All datasets including raw customer data, processed training sets, and test samples.

sagemaker_deployment/ folder - AWS deployment package with model files and sample test CSV.

visualization- Data visualization scripts and graphs for analysis and model performance.

requirements.txt - Python dependencies including XGBoost, SHAP, scikit-learn, and AWS SDK

## Methodology
The project followed a structured approach from problem definition through deployment:
1. Business requirement analysis and success metric definition
2. Data collection and integration from disparate sources
3. Model development with emphasis on interpretability
4. Infrastructure design for scalable deployment
5. Continuous monitoring and optimization framework

## Future Enhancements
- Integration with CRM systems for seamless workflow
- Advanced segmentation for personalized retention strategies
- Real-time dashboard for customer health monitoring
- A/B testing framework for campaign optimization

## OUTPUT
- The Output of the model is shown in Descriptions_Churn_rate_Analyser Folder
