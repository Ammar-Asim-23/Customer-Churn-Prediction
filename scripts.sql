CREATE DATABASE model_predictions;

USE model_predictions;

CREATE TABLE dataset (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gender VARCHAR(10),
    SeniorCitizen INT,
    Partner VARCHAR(10),
    Dependents VARCHAR(10),
    tenure FLOAT,
    PhoneService VARCHAR(10),
    MultipleLines VARCHAR(20),
    InternetService VARCHAR(20),
    OnlineSecurity VARCHAR(20),
    OnlineBackup VARCHAR(20),
    DeviceProtection VARCHAR(20),
    TechSupport VARCHAR(20),
    StreamingTV VARCHAR(20),
    StreamingMovies VARCHAR(20),
    Contract VARCHAR(20),
    PaperlessBilling VARCHAR(10),
    PaymentMethod VARCHAR(50),
    MonthlyCharges FLOAT,
    TotalCharges FLOAT
);

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT,
    model_name VARCHAR(50),
    prediction VARCHAR(10),
    FOREIGN KEY (dataset_id) REFERENCES dataset(id)
);
