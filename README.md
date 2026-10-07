# 💳 Credit Card Fraud Detection

A Machine Learning based web application for detecting potentially fraudulent credit card transactions using **Random Forest** and **SMOTE**.

## 🚀 Live Application

🔗 **[Launch Credit Card Fraud Detection App](https://credit-card-fraud-detection-wfxxgkrwyeglmmphc328if.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud** and can be accessed online for credit card fraud detection.

---

## 📌 Project Overview

Credit card fraud is a major challenge in digital financial transactions. This project develops a Machine Learning based system to identify potentially fraudulent credit card transactions.

The application allows users to upload a transaction CSV file and automatically analyzes the transactions to generate fraud detection results through an interactive Streamlit interface.

---

## 🎯 Objectives

- Detect potentially fraudulent credit card transactions
- Handle highly imbalanced transaction data
- Apply SMOTE for class imbalance handling
- Train a Random Forest classification model
- Perform batch fraud prediction
- Visualize transaction and prediction results
- Evaluate model performance using classification metrics and ROC-AUC

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Imbalanced-learn | SMOTE |
| Joblib | Model Serialization |
| Streamlit | Web Application |
| Matplotlib | Data Visualization |

---

## 🤖 Machine Learning Model

### Random Forest

The project uses a **Random Forest Classifier** to classify credit card transactions as legitimate or potentially fraudulent.

Random Forest combines multiple decision trees to improve classification performance and provide a robust prediction model.

### ⚖️ Class Imbalance Handling

Credit card fraud datasets are highly imbalanced because legitimate transactions significantly outnumber fraudulent transactions.

**SMOTE (Synthetic Minority Over-sampling Technique)** is used to improve the representation of the minority fraud class during model training.

---

## 🔄 Machine Learning Pipeline

```text
Transaction Dataset
        ↓
Data Loading
        ↓
Data Preprocessing
        ↓
Class Imbalance Handling
        ↓
SMOTE
        ↓
Random Forest
        ↓
Fraud Prediction
        ↓
Prediction Results
