# 💳 Credit Card Fraud Detection

## 📌 Project Overview

A machine learning-based credit card fraud detection system designed to identify potentially fraudulent transactions using classification techniques.

The project addresses the highly imbalanced nature of credit card transaction data using **SMOTE (Synthetic Minority Over-sampling Technique)** and uses a **Random Forest Classifier** for fraud detection.

The project also includes an interactive **Streamlit web application** for transaction analysis and fraud prediction.

---

## 🎯 Objectives

- Detect fraudulent credit card transactions
- Handle highly imbalanced transaction data
- Apply data preprocessing and feature engineering
- Use SMOTE to improve minority-class detection
- Train a Random Forest classification model
- Evaluate model performance using fraud-focused metrics
- Provide an interactive transaction prediction interface

---

## 🧠 Machine Learning Approach

The project follows the following workflow:

**Data Loading → Data Preprocessing → Feature Engineering → Train-Test Split → SMOTE → Random Forest → Model Evaluation → Fraud Prediction**

### Key Techniques

- Data preprocessing
- Feature engineering
- Log transformation
- Stratified train-test split
- SMOTE for class imbalance
- Random Forest classification
- Fraud probability prediction

---

## ⚖️ Handling Class Imbalance

Credit card fraud datasets are highly imbalanced because fraudulent transactions represent only a small portion of all transactions.

To address this problem, **SMOTE** is applied during model training to generate synthetic samples of the minority class and improve fraud detection capability.

---

## 📊 Model Performance

The trained Random Forest model achieved:

**ROC-AUC: 0.9689**

Additional evaluation metrics include:

- Precision
- Recall
- F1-Score
- Classification Report
- ROC-AUC

---

## 🚀 Streamlit Application

The project includes an interactive Streamlit application that allows users to:

- Upload transaction CSV files
- Analyse transaction data
- Predict fraudulent transactions
- Calculate fraud probability
- View fraudulent vs legitimate transaction counts
- Visualize transaction distribution
- Download prediction results

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Random Forest
- SMOTE
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── models/
│   └── fraud_detection_model.pkl
│
├── notebooks/
│
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── main.py