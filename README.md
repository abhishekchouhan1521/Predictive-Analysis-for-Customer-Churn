# Predictive-Analysis-for-Customer-Churn
# This project aims to predict customer churn for a telecom company using machine learning models. The goal is to predict whether a customer will churn (leave the service) based on various features such as tenure, contract type, payment method, and other demographic information.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('../data/Telco-Customer-Churn.csv')

# Data Preprocessing
# Convert 'TotalCharges' to numeric and handle errors (e.g., non-numeric values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Encode categorical columns using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Define features and target
X = df.drop(columns=['Churn', 'CustomerID'])
y = df['Churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training: Logistic Regression, Random Forest, SVM
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

# Save the best model (Random Forest, for example)
best_model = models['Random Forest']
joblib.dump(best_model, '../churn_model.pkl')

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Convert 'TotalCharges' to numeric and handle errors (non-numeric values)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Partner', 'Dependents', 'PhoneService', 
                           'MultipleLines', 'InternetService', 'OnlineSecurity']

    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(X_train, y_train, X_test, y_test):
    # Train different models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }
    
    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy}")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    # Save the best model
    joblib.dump(best_model, '../churn_model.pkl')

    return best_model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate_model(y_test, y_pred):
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Churn", "Churn"], yticklabels=["Not Churn", "Churn"])
    plt.title("Confusion Matrix")
    plt.show()

def load_model():
    # Load the saved model
    model = joblib.load('../churn_model.pkl')
    return model

