import streamlit as st
import pickle

import numpy as np
import pandas as pd

df_1=pd.read_csv("first_telc.csv")

model = pickle.load(open("logistics_regression.sav", "rb")) 

st.title("Churn Prediction Model")
#SeniorCitizen,MonthlyCharges,TotalCharges,gender,Partner,Dependents,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,tenure

seniorCitizen = st.selectbox("Senior citizen? . Choose 0 for No, Choose 1 for Ys", [0,1])
monthlyCharges = st.number_input("Input Monthly Charges")
totalCharges  = st.number_input("Input Total Charges")
gender = st.selectbox("Choose sex", ['Male','Female'])
partner = st.select_slider("Has a partner? ", ['Yes','No'])
dependents = st.select_slider("Has dependents? ", ['Yes','No'])
phoneService = st.select_slider("Has phone service? ", ['Yes','No'])
multipleLines = st.selectbox("Has multiple lines? ", ['Yes','No','No phone service'])
internetService = st.selectbox("Choose internet service ", ['DSL','Fiber optic','No phone service'])
onlineSecurity = st.selectbox("Has online security? ", ['Yes','No','No internet service'])
onlineBackup = st.selectbox("Has online backup? ", ['Yes','No','No internet service'])
deviceProtection = st.selectbox("Has device protection? ", ['Yes','No','No internet service'])
techSupport = st.selectbox("Has tech support? ", ['Yes','No','No internet service'])
streamingTV = st.selectbox("Has Streaming TV? ", ['Yes','No','No internet service'])
streamingMovies = st.selectbox("Streams movies? ", ['Yes','No','No internet service'])
contract = st.selectbox("Select Contract type ", ['Month-to-month','One year','Two year'])
paperlessBilling = st.select_slider("Paperless Billing? ", ['Yes','No'])
paymentMethod = st.selectbox("Select payment method ", ['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
tenure = st.number_input("Input tenure (in months)")

def predict(): 
    row = np.array([seniorCitizen,monthlyCharges,totalCharges,gender,partner,dependents,phoneService,multipleLines,internetService,onlineSecurity,onlineBackup,deviceProtection,techSupport,streamingTV,streamingMovies,contract,paperlessBilling,paymentMethod,tenure]) 
    new_df = pd.DataFrame([row], columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure'])
    X = pd.concat([df_1, new_df], ignore_index = True) 
        # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce')
    
    X['tenure_group'] = pd.cut(X.tenure, range(1, 80, 12), right=False, labels=labels)
    #drop column customerID and tenure
    X.drop(columns= ['tenure'], axis=1, inplace=True)   
    
    new_df_dummies = pd.get_dummies(X[['gender','SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']]).astype(int)
    print(new_df_dummies.columns.values)

    prediction = model.predict(new_df_dummies.values)
    #probablity = model.predict_proba(X)[:,1]
    if prediction[0] == 1: 
        st.error(f'Customer will churn.')
        
    else: 
        st.success('Customer will not churn:thumbsup:') 
        

trigger = st.button('Predict', on_click=predict)
