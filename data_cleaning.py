import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#Create a dataframe
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
file = r"C:\Users\shre0\vs_code sign\Churn_rate_Analyser\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = load_data(file)
#print(df.columns.values)
#print(df.info())
df=df.drop(['customerID'],axis=1)
#print(df.head())
#values that cannot be parsed are converted to nan
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
#print(df.isnull().sum())
#Handling missing values--filled with median
df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)
#dropping rows wher tenure is 0
df.drop(df[df['tenure'] == 0].index, axis=0,inplace=True)
#print(df[df['tenure'] == 0].index)
#Converting Yes/No to 1/0
df["SeniorCitizen"]=df['SeniorCitizen'].replace({1:'Yes',0:"No"})
df["Churn"]=df['Churn'].replace({'Yes':1,"No":0})