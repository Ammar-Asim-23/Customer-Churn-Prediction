# Customer Retention Prediction Project

## Overview
This project involves data analysis and machine learning model training for predicting customer retention from a churn dataset. The project utilizes a Jupyter notebook for exploratory data analysis (EDA) and model training, and a Python script for deploying the model in a Streamlit app.

## Repository Contents
- `analysis.ipynb`: Jupyter notebook containing exploratory data analysis.
- `training.ipynb`: Jupyter notebook containing model training.
- `app.py`: Python script for deploying the model using Streamlit.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of required Python packages.
- `cleaned_dataset.csv`: The cleaned dataset used for training the models.
- `models/`: Directory where the trained model pickle files are stored.

## Requirements
To run the notebooks and scripts, you need the following dependencies:
- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- streamlit
- jupyter

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Usage

### Data Analysis and Model Training
1. Open the Jupyter notebook `analysis.ipynb`.
2. Follow the steps to load the data, perform exploratory data analysis (EDA), and train the machine learning models.

### Streamlit App
1. Run the Streamlit app using the following command:
```bash
streamlit run app.py
```
2. Open the provided URL in your web browser to interact with the app.

## Setting Up MySQL Database
To store user credentials securely, create a MySQL database and store the following details in a `.env` file:
- `username`
- `password`
- `host`
- `database name`

Here is an example of what the `.env` file should look like:
```dotenv
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_HOST=your_host
DB_NAME=your_database_name
```

Make sure to replace the placeholder values with your actual MySQL credentials.

## Model Evaluation Summary

The project evaluates several machine learning models for predicting customer retention from a churn dataset. The models are assessed based on their test accuracy, F1 score, and ROC AUC score. Below are the results for each model:

- **Random Forest**
  - Test Accuracy: 95.1%
  - F1 Score: 0.957
  - ROC AUC Score: 0.949

- **Gradient Boosting**
  - Test Accuracy: 95.5%
  - F1 Score: 0.96
  - ROC AUC Score: 0.953

- **Support Vector Machine**
  - Test Accuracy: 96.3%
  - F1 Score: 0.966
  - ROC AUC Score: 0.966

- **Logistic Regression**
  - Test Accuracy: 91.6%
  - F1 Score: 0.925
  - ROC AUC Score: 0.916

- **K-Nearest Neighbors**
  - Test Accuracy: 98.0%
  - F1 Score: 0.982
  - ROC AUC Score: 0.978

- **Decision Tree**
  - Test Accuracy: 93.0%
  - F1 Score: 0.938
  - ROC AUC Score: 0.93

- **Ada Boost**
  - Test Accuracy: 93.9%
  - F1 Score: 0.946
  - ROC AUC Score: 0.937

- **XG Boost**
  - Test Accuracy: 95.3%
  - F1 Score: 0.959
  - ROC AUC Score: 0.952

- **Naive Bayes**
  - Test Accuracy: 89.8%
  - F1 Score: 0.908
  - ROC AUC Score: 0.899

### Best Model

The **K-Nearest Neighbors (KNN)** model achieved the highest performance among the evaluated models with the following metrics:
- **Test Accuracy:** 98.0%
- **F1 Score:** 0.982
- **ROC AUC Score:** 0.978

The best model pipeline configuration:
```python
KNeighborsClassifier(n_neighbors=3, weights='distance')
``` 

## Feature Importance for Different Models

**Random Forest**

```
| Feature           | Importance |
|-------------------|------------|
| Contract          | 0.199789   |
| tenure            | 0.180900   |
| MonthlyCharges    | 0.101548   |
| OnlineSecurity    | 0.100847   |
| TotalCharges      | 0.100756   |
| TechSupport       | 0.085109   |
| InternetService   | 0.057598   |
| OnlineBackup      | 0.039588   |
| DeviceProtection  | 0.028089   |
| PaymentMethod     | 0.018662   |
| Dependents        | 0.015825   |
| Partner           | 0.015394   |
| StreamingTV       | 0.010856   |
| StreamingMovies   | 0.010071   |
| gender            | 0.009854   |
| MultipleLines     | 0.008600   |
| PaperlessBilling  | 0.007461   |
| SeniorCitizen     | 0.005519   |
| PhoneService      | 0.003535   |
```

**Gradient Boosting**

```
| Feature           | Importance |
|-------------------|------------|
| Contract          | 0.580647   |
| tenure            | 0.121202   |
| MonthlyCharges    | 0.090292   |
| TotalCharges      | 0.085427   |
| InternetService   | 0.067164   |
| TechSupport       | 0.013331   |
| OnlineSecurity    | 0.009196   |
| OnlineBackup      | 0.007528   |
| PaymentMethod     | 0.005179   |
| Dependents        | 0.004340   |
| MultipleLines     | 0.002619   |
| SeniorCitizen     | 0.002366   |
| DeviceProtection  | 0.002172   |
| PaperlessBilling  | 0.002039   |
| StreamingTV       | 0.002028   |
| gender            | 0.001719   |
| Partner           | 0.001535   |
| PhoneService      | 0.000719   |
| StreamingMovies   | 0.000497   |
```

**Support Vector Machine**

```
| Feature           | Importance |
|-------------------|------------|
| TotalCharges      | 0.447549   |
| tenure            | 0.377472   |
| MonthlyCharges    | 0.371367   |
| Contract          | 0.002236   |
| StreamingTV       | 0.001376   |
| PaymentMethod     | 0.001204   |
| StreamingMovies   | 0.000688   |
| OnlineSecurity    | 0.000688   |
| PaperlessBilling  | 0.000602   |
| MultipleLines     | 0.000516   |
| DeviceProtection  | 0.000516   |
| gender            | 0.000430   |
| OnlineBackup      | 0.000430   |
| Dependents        | 0.000172   |
| SeniorCitizen     | 0.000000   |
| Partner           | 0.000000   |
| PhoneService      | 0.000000   |
| InternetService   | 0.000000   |
| TechSupport       | -0.000258  |
```

**Logistic Regression**

```
| Feature           | Importance |
|-------------------|------------|
| InternetService   | 0.456468   |
| PaperlessBilling  | 0.259925   |
| PaymentMethod     | 0.069234   |
| MonthlyCharges    | 0.061924   |
| TotalCharges      | 0.001223   |
| StreamingMovies   | -0.058378  |
| StreamingTV       | -0.085999  |
| tenure            | -0.187122  |
| MultipleLines     | -0.216107  |
| SeniorCitizen     | -0.238588  |
| PhoneService      | -0.280188  |
| DeviceProtection  | -0.360974  |
| Partner           | -0.419990  |
| gender            | -0.443873  |
| OnlineBackup      | -0.510363  |
| OnlineSecurity    | -0.511429  |
| Dependents        | -0.680224  |
| TechSupport       | -0.763374  |
| Contract          | -1.205329  |
```

**K-Nearest Neighbors**

```
| Feature           | Importance |
|-------------------|------------|
| MonthlyCharges    | 0.296733   |
| TotalCharges      | 0.279966   |
| tenure            | 0.071281   |
| PaymentMethod     | 0.001720   |
| Contract          | 0.000946   |
| PaperlessBilling  | 0.000086   |
| Partner           | 0.000000   |
| DeviceProtection  | 0.000000   |
| OnlineBackup      | 0.000000   |
| PhoneService      | 0.000000   |
| Dependents        | 0.000000   |
| MultipleLines     | 0.000000   |
| StreamingMovies   | 0.000000   |
| OnlineSecurity    | 0.000000   |
| InternetService   | 0.000000   |
| TechSupport       | 0.000000   |
| StreamingTV       | 0.000000   |
| gender            | -0.000430  |
| SeniorCitizen     | -0.000688  |
```

**Decision Tree**

```
| Feature           | Importance |
|-------------------|------------|
| Contract          | 0.534657   |
| TotalCharges      | 0.119488   |
| MonthlyCharges    | 0.107362   |
| tenure            | 0.080831   |
| InternetService   | 0.070151   |
| PaymentMethod     | 0.014998   |
| TechSupport       | 0.010704   |
| OnlineSecurity    | 0.009869   |
| gender            | 0.009167   |
| Dependents        | 0.007443   |
| SeniorCitizen     | 0.005915   |
| OnlineBackup      | 0.005581   |
| MultipleLines     | 0.005290   |
| PhoneService      | 0.004329   |
| PaperlessBilling  | 0.003639   |
| DeviceProtection  | 0.003430   |
| StreamingTV       | 0.002898   |
| Partner           | 0.002408   |
| StreamingMovies   | 0.001840   |
```

**AdaBoost**

```
| Feature           | Importance |
|-------------------|------------|
| TotalCharges      | 0.255      |
| MonthlyCharges    | 0.245      |
| tenure            | 0.175      |
| TechSupport       | 0.055      |
| OnlineSecurity    | 0.055      |
| OnlineBackup      | 0.025      |
| DeviceProtection  | 0.025      |
| PhoneService      | 0.025      |
| Contract          | 0.020      |
| InternetService   | 0.020      |
| StreamingMovies   | 0.020      |
| StreamingTV       | 0.020      |
| MultipleLines     | 0.015      |
| gender            | 0.015      |
| SeniorCitizen     | 0.010      |
| Dependents        | 0.010      |
| Partner           | 0.005      |
| PaymentMethod     | 0.005      |
| PaperlessBilling  | 0.000      |
```

**XGBoost**

```
| Feature           | Importance |
|-------------------|------------|
| Contract          | 0.654443   |
| InternetService   | 0.130281   |
| TechSupport       | 0.023324   |
| Dependents        | 0.023067   |
| tenure            | 0.022033   |
| OnlineSecurity    | 0.017877   |
| MonthlyCharges    | 0.016280   |
| OnlineBackup      | 0.016168   |
| DeviceProtection  | 0.011127   |
| TotalCharges      | 0.010829   |
| PhoneService      | 0.009766   |
| SeniorCitizen     | 0.009382   |
| StreamingMovies   | 0.008740   |
| gender            | 0.008523   |
| MultipleLines     | 0.008490   |
| Partner           | 0.008328   |
| StreamingTV       | 0.007590   |
| PaperlessBilling  | 0.007140   |
| PaymentMethod     | 0.006612   |
```

**Naive Bayes**

```
| Feature           | Importance |
|-------------------|------------|
| Contract          | 0.126913   |
| MonthlyCharges    | 0.019518   |
| OnlineSecurity    | 0.018143   |
| InternetService   | 0.016079   |
| tenure            | 0.015821   |
| TechSupport       | 0.013500   |
| Dependents        | 0.012726   |
| OnlineBackup      | 0.010920   |
| DeviceProtection  | 0.010920   |
| PaymentMethod     | 0.010920   |
| PaperlessBilling  | 0.009871   |
| gender            | 0.009091   |
| Partner           | 0.008223   |
| StreamingMovies   | 0.007364   |
| StreamingTV       | 0.006519   |
| PhoneService      | 0.005683   |
| MultipleLines     | 0.004854   |
| SeniorCitizen     | 0.004032   |
| TotalCharges      | 0.003217   |
```

## Model Performance Comparison


### Classification Report for Random Forest:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.94      0.96      0.95       528
           1       0.97      0.95      0.96       653

    accuracy                           0.95      1181
    macro avg       0.95      0.95      0.95      1181
    weighted avg       0.95      0.95      0.95      1181

-----------------------------------------------------


### Classification Report for Gradient Boosting:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.94      0.95      0.94       528
           1       0.96      0.95      0.95       653

    accuracy                           0.95      1181
    macro avg       0.95      0.95      0.95      1181
    weighted avg       0.95      0.95      0.95      1181

-----------------------------------------------------


### Classification Report for Support Vector Machine:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.93      0.99      0.96       528
           1       0.99      0.94      0.96       653

    accuracy                           0.96      1181
    macro avg       0.96      0.96      0.96      1181
    weighted avg       0.96      0.96      0.96      1181

-----------------------------------------------------


### Classification Report for Logistic Regression:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.85      0.93      0.89       528
           1       0.94      0.87      0.90       653

    accuracy                           0.90      1181
    macro avg       0.90      0.90      0.90      1181
    weighted avg       0.90      0.90      0.90      1181

-----------------------------------------------------


### Classification Report for K-Nearest Neighbors:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.99      0.97      0.98       528
           1       0.98      0.99      0.98       653

    accuracy                           0.98      1181
    macro avg       0.98      0.98      0.98      1181
    weighted avg       0.98      0.98      0.98      1181

-----------------------------------------------------


### Classification Report for Decision Tree:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.91      0.92      0.92       528
           1       0.94      0.93      0.93       653

    accuracy                           0.92      1181
    macro avg       0.92      0.92      0.92      1181
    weighted avg       0.92      0.92      0.92      1181

-----------------------------------------------------


### Classification Report for Ada Boost:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.92      0.93      0.92       528
           1       0.94      0.94      0.94       653

    accuracy                           0.93      1181
    macro avg       0.93      0.93      0.93      1181
    weighted avg       0.93      0.93      0.93      1181

-----------------------------------------------------


### Classification Report for XG Boost:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.94      0.96      0.95       528
           1       0.96      0.95      0.96       653

    accuracy                           0.96      1181
    macro avg       0.95      0.96      0.95      1181
    weighted avg       0.96      0.96      0.96      1181

-----------------------------------------------------


### Classification Report for Naive Bayes:
-----------------------------------------------------
              precision    recall  f1-score   support

           0       0.83      0.91      0.87       528
           1       0.93      0.85      0.89       653

    accuracy                           0.88      1181
    macro avg       0.88      0.88      0.88      1181
    weighted avg       0.88      0.88      0.88      1181

-----------------------------------------------------


## Contributing
Contributions are welcome! Please create a pull request with a clear description of your changes.
