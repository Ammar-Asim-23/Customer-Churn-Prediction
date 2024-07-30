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

## Project Structure
- `analysis.ipynb`: Contains all the data analysis and model training steps.
- `app.py`: Streamlit app for deploying the model and predicting customer retention.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of required Python packages.
- `models/`: Directory to save trained model pickle files.
- `cleaned_dataset.csv`: The dataset used for training.

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

## Contributing
Contributions are welcome! Please create a pull request with a clear description of your changes.
