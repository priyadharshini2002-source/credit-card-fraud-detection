# 💳 Credit Card Fraud Detection

A Machine Learning based web application for detecting potentially fraudulent credit card transactions using **Random Forest** and **SMOTE**.

---

## 🚀 Live Application

🔗 **[Launch Credit Card Fraud Detection App](https://credit-card-fraud-detection-wfxxgkrwyeglmmphc328if.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud** and is available online for credit card fraud detection.

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
| Imbalanced-learn | SMOTE and Class Imbalance Handling |
| Joblib | Model Serialization |
| Streamlit | Web Application |
| Matplotlib | Data Visualization |
| GitHub | Version Control and Repository |
| Streamlit Community Cloud | Cloud Deployment |

---

## 🤖 Machine Learning Model

### 🌳 Random Forest

The project uses a **Random Forest Classifier** to classify credit card transactions as legitimate or potentially fraudulent.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance and robustness.

### ⚖️ SMOTE

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
        ↓
Data Visualization

✨ Key Features
- ✅ Machine Learning based fraud detection
- ✅ Random Forest classification
- ✅ SMOTE for class imbalance handling
- ✅ CSV transaction file upload
- ✅ Uploaded transaction data preview
- ✅ Batch transaction prediction
- ✅ Fraud prediction results
- ✅ Prediction summary
- ✅ Transaction data visualization
- ✅ ROC-AUC evaluation
- ✅ Interactive Streamlit interface
- ✅ Cloud deployment
- ✅ Online live demo


## 📊 Model Performance

| Metric | Score |
|---|---:|
| ROC-AUC | **0.9689** |
| Fraud Precision | **0.84** |
| Fraud Recall | **0.83** |
| Fraud F1-Score | **0.83** |

### 📈 Performance Metrics

| Metric | Description |
|---|---|
| Precision | Measures how many predicted fraud transactions were actually fraudulent |
| Recall | Measures how many actual fraud transactions were successfully detected |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Measures the model's ability to distinguish between legitimate and fraudulent transactions |

---

# 💻 Streamlit Application

The project includes an interactive Streamlit web application for uploading transaction data, detecting fraudulent transactions, viewing prediction results, and analyzing transaction data.

---

## 🏠 Home Page

The Home page provides an overview of the Credit Card Fraud Detection system, machine learning model, class imbalance handling technique, model performance, and the complete machine learning pipeline.

![Home Page](https://github.com/priyadharshini2002-source/credit-card-fraud-detection/blob/main/home_page.png?raw=true)

---

## 📤 Uploaded Transaction Data

Users can upload a credit card transaction CSV file through the Streamlit application. The uploaded transaction data is displayed for further analysis and fraud detection.

![Uploaded Transaction Data](https://github.com/priyadharshini2002-source/credit-card-fraud-detection/blob/main/uploaded_data.png?raw=true)

---

## 🔍 Fraud Transaction Prediction

The application performs fraud detection on the uploaded transaction dataset using the trained Random Forest model.

![Fraud Transaction Prediction](https://github.com/priyadharshini2002-source/credit-card-fraud-detection/blob/main/fraud_prediction.png?raw=true)

---

## 📊 Prediction Summary

The Prediction Summary section displays the generated fraud detection results and provides an overview of the prediction output.

![prediction summary](https://github.com/priyadharshini2002-source/credit-card-fraud-detection/blob/main/prediction_summary.png?raw=true)

---

## 📈 Transaction Analysis

The Analysis section provides visual insights into the transaction data and helps understand the distribution of legitimate and potentially fraudulent transactions.

![Transaction Analysis](https://github.com/priyadharshini2002-source/credit-card-fraud-detection/blob/main/analysis_completion.png?raw=true)

---

## 📂 Project Structure

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

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/priyadharshini2002-source/credit-card-fraud-detection.git
```

### 2. Navigate to the Project Directory

```bash
cd credit-card-fraud-detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📁 Dataset

The project uses a credit card transaction dataset containing legitimate and fraudulent transactions.

### Dataset Features

| Feature | Description |
|---|---|
| `Time` | Time elapsed between transactions |
| `V1` – `V28` | Anonymized transaction features |
| `Amount` | Transaction amount |
| `Class` | Target variable |

### Target Variable

| Class | Meaning |
|---:|---|
| `0` | Legitimate Transaction |
| `1` | Fraudulent Transaction |

---

## 🔐 Prediction Output

The application classifies transactions into:

- **Legitimate Transaction**
- **Potentially Fraudulent Transaction**

The prediction results can also be exported as a CSV file for further analysis.

---

## 🔮 Future Enhancements

- Real-time transaction monitoring
- Fraud probability scoring
- Advanced anomaly detection
- XGBoost and LightGBM model comparison
- Real-time API integration
- Explainable AI using SHAP
- Automated fraud alerts
- Continuous model retraining
- Real-time fraud detection dashboard

---

## 👩‍💻 Author

**S. Priyadharshini**

MSc Data Science

---

## 📜 License

This project is developed for educational and portfolio purposes.
```

