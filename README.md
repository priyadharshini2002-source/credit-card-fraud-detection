# 💳 Credit Card Fraud Detection

A machine learning-based web application for detecting potentially fraudulent credit card transactions using **Random Forest** and **SMOTE (Synthetic Minority Over-sampling Technique)**.

The application provides an interactive **Streamlit interface** where users can upload transaction data, perform batch fraud detection, analyze prediction results, and visualize transaction patterns.

### 🚀 Live Demo

**[Launch Credit Card Fraud Detection App](https://credit-card-fraud-detection-wfxxgkrwyeglmmphc328if.streamlit.app/)**

---

## 📌 Project Overview

Credit card fraud is a major challenge in digital financial transactions due to the highly imbalanced nature of transaction data, where legitimate transactions significantly outnumber fraudulent transactions.

This project develops an end-to-end machine learning solution to identify potentially fraudulent transactions.

The system combines:

- **Data preprocessing**
- **SMOTE-based class imbalance handling**
- **Random Forest classification**
- **Batch transaction prediction**
- **Fraud prediction analysis**
- **ROC-AUC evaluation**
- **Interactive data visualization**
- **Streamlit web deployment**

Users can upload a CSV file containing transaction data, and the application automatically processes the data and generates fraud detection results.

---

## 🎯 Objectives

The main objectives of this project are:

- Detect potentially fraudulent credit card transactions.
- Handle highly imbalanced transaction datasets.
- Apply **SMOTE** to improve minority-class representation during training.
- Train a **Random Forest Classifier** for fraud detection.
- Perform batch prediction on transaction datasets.
- Provide an interactive web-based fraud detection interface.
- Visualize transaction and prediction results.
- Evaluate the model using classification metrics and **ROC-AUC**.
- Deploy the machine learning application online using Streamlit Community Cloud.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 Random Forest | Ensemble machine learning model for fraud classification |
| ⚖️ SMOTE | Handles class imbalance by generating synthetic minority samples |
| 📤 CSV Upload | Allows users to upload transaction datasets |
| 👀 Data Preview | Displays uploaded transaction records |
| 🔍 Fraud Detection | Classifies transactions as legitimate or potentially fraudulent |
| 📊 Prediction Summary | Provides an overview of prediction results |
| 📈 Data Visualization | Visualizes transaction and fraud distributions |
| 📋 Batch Prediction | Processes multiple transactions simultaneously |
| 📥 CSV Export | Allows prediction results to be exported |
| 📐 ROC-AUC | Evaluates the model's classification performance |
| 🌐 Streamlit | Provides an interactive web interface |
| ☁️ Cloud Deployment | Application deployed using Streamlit Community Cloud |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Programming language |
| **Pandas** | Data processing and manipulation |
| **NumPy** | Numerical computation |
| **Scikit-learn** | Machine learning and model evaluation |
| **Imbalanced-learn** | SMOTE and class imbalance handling |
| **Joblib** | Model serialization |
| **Streamlit** | Interactive web application |
| **Matplotlib** | Data visualization |
| **GitHub** | Version control and source code management |
| **Streamlit Community Cloud** | Application deployment |

---

# 🤖 Machine Learning Approach

## 🌳 Random Forest Classifier

The project uses a **Random Forest Classifier** to classify credit card transactions into legitimate and potentially fraudulent transactions.

Random Forest is an ensemble learning algorithm that combines multiple decision trees and aggregates their predictions to improve classification performance and robustness.

### Why Random Forest?

- Handles nonlinear relationships effectively.
- Works well with high-dimensional datasets.
- Provides robust classification performance.
- Reduces overfitting compared with a single decision tree.
- Can handle complex feature interactions.

---

## ⚖️ SMOTE for Class Imbalance

Credit card fraud datasets are typically highly imbalanced because legitimate transactions greatly outnumber fraudulent transactions.

If the model is trained directly on such data, it may become biased toward the majority class.

This project uses **SMOTE (Synthetic Minority Over-sampling Technique)** during model training.

SMOTE generates synthetic examples of the minority fraud class to provide the model with a more balanced training dataset.

### Important Note

SMOTE is applied to the **training data**, not directly to the test data. This helps avoid data leakage and ensures that model evaluation remains meaningful.

---

# 🔄 Machine Learning Pipeline

```text
                 Transaction Dataset
                         │
                         ▼
                  Data Loading
                         │
                         ▼
                 Data Preprocessing
                         │
                         ▼
               Train / Test Split
                         │
                         ▼
              Class Imbalance Handling
                         │
                         ▼
                       SMOTE
                         │
                         ▼
                Random Forest Model
                         │
                         ▼
                 Model Evaluation
                         │
                         ▼
                 Fraud Prediction
                         │
                         ▼
              Prediction & Analysis
                         │
                         ▼
              Streamlit Visualization
```

---

# 📊 Model Evaluation

The model performance can be evaluated using multiple classification metrics:

### Accuracy

Measures the overall percentage of correctly classified transactions.

### Precision

Measures how many transactions predicted as fraudulent are actually fraudulent.

### Recall

Measures how many actual fraudulent transactions are correctly identified.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between legitimate and fraudulent transactions across different classification thresholds.

For fraud detection, **Precision, Recall, F1-Score, and ROC-AUC** are particularly important because accuracy alone can be misleading when the dataset is highly imbalanced.

---

# 💻 Streamlit Application

## 🏠 Home Page

The Home page introduces the Credit Card Fraud Detection system and provides information about:

- Machine learning model
- SMOTE-based imbalance handling
- Model evaluation
- Fraud detection workflow
- Application features

![Home Page](https://raw.githubusercontent.com/priyadharshini2002-source/credit-card-fraud-detection/main/home_page.png)

---

## 📤 Uploaded Transaction Data

Users can upload a credit card transaction CSV file through the Streamlit application.

The uploaded dataset is displayed in the application for inspection before performing fraud detection.

![Uploaded Transaction Data](https://raw.githubusercontent.com/priyadharshini2002-source/credit-card-fraud-detection/main/uploaded_data.png)

---

## 🔍 Fraud Transaction Prediction

The application processes the uploaded transaction data and uses the trained **Random Forest model** to classify transactions.

The prediction output identifies transactions as:

- **Legitimate Transaction**
- **Potentially Fraudulent Transaction**

![Fraud Transaction Prediction](https://raw.githubusercontent.com/priyadharshini2002-source/credit-card-fraud-detection/main/fraud_prediction.png)

---

## 📊 Prediction Summary

The Prediction Summary section provides an overview of the generated predictions and helps users understand the number and distribution of detected transactions.

![Prediction Summary](https://raw.githubusercontent.com/priyadharshini2002-source/credit-card-fraud-detection/main/prediction_summary.png)

---

## 📈 Transaction Analysis

The Analysis section provides visual insights into the transaction dataset and prediction results.

It helps users understand the distribution of legitimate and potentially fraudulent transactions.

![Transaction Analysis](https://raw.githubusercontent.com/priyadharshini2002-source/credit-card-fraud-detection/main/analysis_completion.png)

---

# 📁 Dataset

The project uses a credit card transaction dataset containing both legitimate and fraudulent transactions.

### Dataset Features

| Feature | Description |
|---|---|
| **Time** | Time elapsed between transactions |
| **V1 – V28** | Anonymized transaction features |
| **Amount** | Transaction amount |
| **Class** | Target variable |

### Target Variable

| Class | Meaning |
|---:|---|
| **0** | Legitimate Transaction |
| **1** | Fraudulent Transaction |

> **Note:** The dataset contains anonymized features, which is common in publicly available credit card fraud detection datasets.

---

# 📥 Prediction Output

The application classifies transactions into two categories:

```text
0 → Legitimate Transaction
1 → Potentially Fraudulent Transaction
```

The prediction results can also be exported as a CSV file for further analysis.

---

# 📂 Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── main.py
├── creditcard.csv
├── requirements.txt
├── README.md
├── fraud_prediction_results.csv
│
├── home_page.png
├── uploaded_data.png
├── fraud_prediction.png
├── prediction_summary.png
├── analysis_completion.png
│
├── models/
│   └── fraud_detection_model.pkl
│
└── src/
    ├── model.py
    ├── preprocessing.py
    ├── predict.py
    └── train_model.py
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/priyadharshini2002-source/credit-card-fraud-detection.git
```

## 2. Navigate to the Project Directory

```bash
cd credit-card-fraud-detection
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

---

# 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### 🔗 Live Application

**[Open Credit Card Fraud Detection App](https://credit-card-fraud-detection-wfxxgkrwyeglmmphc328if.streamlit.app/)**

The deployed application allows users to interact with the fraud detection system directly through a web browser without installing the project locally.

---

# 🔮 Future Enhancements

The following improvements can be incorporated in future versions:

- 🔴 Real-time transaction monitoring
- 📊 Fraud probability scoring
- 🤖 XGBoost and LightGBM model comparison
- 🔎 Advanced anomaly detection
- 🧠 Explainable AI using SHAP
- 🚨 Automated fraud alerts
- 🔄 Continuous model retraining
- 🌐 Real-time fraud detection API
- 📈 Advanced fraud monitoring dashboard
- ☁️ Scalable cloud-based deployment

---

# 🔐 Limitations

This project is developed primarily for **educational and portfolio purposes**.

The model should not be directly used for real-world financial decision-making without additional validation, security controls, monitoring, and domain-specific testing.

Real-world fraud detection systems may require:

- Real-time transaction streams
- Extremely low-latency prediction
- Advanced anomaly detection
- Continuous model monitoring
- Concept-drift detection
- Explainability
- Security and privacy controls
- Human fraud investigation workflows

---

# 👩‍💻 Author

**S. Priyadharshini**

MSc Data Science

**Skills:**  
Machine Learning • Data Analytics • Python • Scikit-learn • Streamlit • Data Visualization

---

# 📜 License

This project is developed for **educational and portfolio purposes**.
