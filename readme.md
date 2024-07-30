# Customer Retention Prediction Project

## Overview
This project involves data analysis and machine learning model training for predicting customer retention from a churn dataset. The project utilizes a Jupyter notebook for exploratory data analysis (EDA) and model training, and a Python script for deploying the model in a Streamlit app.

## Repository Contents
- `analysis.ipynb`: Jupyter notebook containing data analysis and model training.
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

## Contributing
Contributions are welcome! Please create a pull request with a clear description of your changes.
