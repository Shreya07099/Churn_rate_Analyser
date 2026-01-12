# Churn_rate_Analyser
Author-Shreya Nair, VIT Vellore
#This Project Aims to :
1.Identify and prioritize at-risk customers within the highest Customer Lifetime Value (CLV) segments with ≥85% precision, enabling targeted retention campaigns that aim to reduce churn in this segment by 20% within one quarter.
2.Surface actionable reasons for churn risk (e.g., "low usage of Feature X," "negative support ticket sentiment," "payment failure pattern") to enable personalized, timely interventions through the most effective channel (e.g., SMS, email, in-app message).
3.Develop a modular data pipeline that automatically ingests, consolidates, and refreshes customer data from multiple sources (transactional, behavioral, support) to create a single source of truth for churn analysis, serving predictions via a reliable API.
4.Build an event-driven AWS pipeline that ingests churn risk scores from the model API, triggers AWS Lambda functions to evaluate rules and personalize messages, and integrates with Amazon Pinpoint or AWS SNS to execute automated, multi-channel re-engagement campaigns via SMS/WhatsApp/Email.
#DataSet used:
#Note: We have the following column values: ['customerID' 'gender' 'SeniorCitizen' 'Partner' 'Dependents' 'tenure'
 'PhoneService' 'MultipleLines' 'InternetService' 'OnlineSecurity'
 'OnlineBackup' 'DeviceProtection' 'TechSupport' 'StreamingTV'
 'StreamingMovies' 'Contract' 'PaperlessBilling' 'PaymentMethod'
 'MonthlyCharges' 'TotalCharges' 'Churn']
