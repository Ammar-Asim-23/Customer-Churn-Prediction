import mysql.connector
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import json
from sklearn import preprocessing
import mysql
import os
from dotenv import load_dotenv
load_dotenv()

# Load encoders
encoder = pickle.load(open('models/encoders.pkl', 'rb'))

# Define all the columns
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 
        'TotalCharges']

def load_model(model_name):
    return pickle.load(open(f'models/{model_name}', 'rb'))
def create_connection():
    connection = mysql.connector.connect(
        host= os.environ.get("host"),
        user=os.environ.get("username"), 
        password=os.environ.get("password"),
        database=os.environ.get("database")
        )
    return connection

def insert_dataset(connection, data):
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO dataset (gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
                         MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
                         DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
                         Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, 
                         TotalCharges)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = tuple(data[0])

    cursor.execute(insert_query, values)
    connection.commit()
    return cursor.lastrowid

def insert_prediction(connection, dataset_id, model_name, prediction):
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO predictions (dataset_id, model_name, prediction)
    VALUES (%s, %s, %s)
    """
    cursor.execute(insert_query, (dataset_id, model_name, prediction))
    connection.commit()

def main():
    st.title("Income Predictor")
    
    # HTML styling for the title
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Model selection with K-Nearest Neighbors as default and indicating it's the best
    model_option = st.selectbox("Select Model", [
        'Ada Boost.pkl', 'Naive Bayes.pkl', 'Support Vector Machine.pkl', 
        'Decision Tree.pkl', 'Gradient Boosting.pkl', 'Logistic Regression.pkl', 
        'Random Forest.pkl', 'XG Boost.pkl', 'K-Nearest Neighbors.pkl (Best)'
    ], index=8)  # Index 8 is for 'K-Nearest Neighbors.pkl (Best)'

    # Load the selected model
    model = load_model(model_option.replace(' (Best)', ''))
    
    # User input
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["0", "1"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.text_input("Tenure", "0")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
    MonthlyCharges = st.text_input("Monthly Charges", "0")
    TotalCharges = st.text_input("Total Charges", "0")

    if st.button("Predict"):
        # Create DataFrame from user input
        data = {
            'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner], 
            'Dependents': [Dependents], 'tenure': [float(tenure)], 'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines], 'InternetService': [InternetService], 
            'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup], 
            'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport], 
            'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies],
            'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling], 
            'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [float(MonthlyCharges)], 
            'TotalCharges': [float(TotalCharges)]
        }
        # Insert dataset into MySQL
        connection = create_connection()
        df = pd.DataFrame(data, columns=cols)

        dataset_id = insert_dataset(connection, df.values)
        
        # Apply the mappings for categorical variables
        non_cat_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        enc_cols = df.drop(non_cat_cols, axis=1).columns

        # Encode categorical columns
        for col in enc_cols:
            if col in encoder.keys():
                le = encoder[col]
                df[col] = le.transform(df[col])
        
        # Ensure the DataFrame has the correct order of columns
        df = df[cols]


        


        
        # Make prediction
        features_list = df.values.tolist()
        prediction = model.predict(features_list)
    
        output = int(prediction[0])
        result = "Yes" if output == 1 else "No"

        # Insert prediction into MySQL
        insert_prediction(connection, dataset_id, model_option.replace(' (Best)', ''), result)

        st.success(f'Prediction: {result}')
        connection.close()

if __name__ == '__main__':
    main()
