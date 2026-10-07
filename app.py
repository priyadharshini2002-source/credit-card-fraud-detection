import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------

MODEL_PATH = "models/fraud_detection_model.pkl"
ROC_AUC = 0.9689

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("💳 Fraud Detection System")
st.sidebar.markdown(
    "Machine Learning based credit card fraud detection."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Fraud Prediction"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.title("💳 Credit Card Fraud Detection System")

    st.markdown(
        """
        ### Machine Learning Based Fraud Detection

        This application uses a **Random Forest classifier**
        with **SMOTE** to identify potentially fraudulent
        credit card transactions.
        """
    )

    st.divider()

    # Model Summary
    st.subheader("📊 Model Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🤖 Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "⚖️ Imbalance Handling",
            "SMOTE"
        )

    with col3:
        st.metric(
            "📈 ROC-AUC",
            "0.9689"
        )

    with col4:
        st.metric(
            "🎯 Fraud F1-Score",
            "0.83"
        )

    st.divider()

    # Project Overview
    st.subheader("📌 Project Overview")

    st.write(
        """
        Credit card fraud detection is a highly imbalanced
        classification problem where fraudulent transactions
        represent only a small portion of total transactions.

        This project applies data preprocessing, SMOTE-based
        class balancing and Random Forest classification to
        identify suspicious transactions.
        """
    )

    # Workflow
    st.subheader("🔄 Machine Learning Workflow")

    workflow = [
        "📂 Data Loading",
        "🔧 Data Preprocessing",
        "⚖️ SMOTE",
        "🌳 Random Forest",
        "📊 Model Evaluation",
        "🚨 Fraud Prediction"
    ]

    cols = st.columns(len(workflow))

    for col, step in zip(cols, workflow):
        with col:
            st.info(step)

    st.divider()

    # Key Features
    st.subheader("✨ Application Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            ✅ CSV transaction upload

            ✅ Fraudulent transaction detection

            ✅ Fraud probability calculation
            """
        )

    with feature_col2:
        st.markdown(
            """
            ✅ Transaction distribution analysis

            ✅ Prediction result table

            ✅ Downloadable prediction results
            """
        )

    st.divider()

    st.info(
        "👉 Select **🔍 Fraud Prediction** from the sidebar "
        "to analyse transaction data."
    )


# --------------------------------------------------
# FRAUD PREDICTION PAGE
# --------------------------------------------------

elif page == "🔍 Fraud Prediction":

    st.title("🔍 Fraud Transaction Prediction")

    st.markdown(
        """
        Upload a transaction CSV file to classify transactions
        as **Legitimate** or **Potentially Fraudulent**.
        """
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "📁 Please upload a CSV file containing the same "
            "features used during model training."
        )

    else:

        try:

            # ----------------------------------------------
            # Load Data
            # ----------------------------------------------

            data = pd.read_csv(uploaded_file)

            st.success(
                f"✅ File uploaded successfully — "
                f"{len(data):,} transactions found."
            )

            # ----------------------------------------------
            # Dataset Preview
            # ----------------------------------------------

            st.subheader("📄 Dataset Preview")

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

            st.divider()

            # ----------------------------------------------
            # Prepare Prediction Data
            # ----------------------------------------------

            prediction_data = data.copy()

            # Remove target column if present
            if "Class" in prediction_data.columns:
                prediction_data = prediction_data.drop(
                    "Class",
                    axis=1
                )

            # Apply same preprocessing used during training
            if "Amount" in prediction_data.columns:

                prediction_data["Amount"] = np.log1p(
                    prediction_data["Amount"]
                )

            # ----------------------------------------------
            # Prediction
            # ----------------------------------------------

            predictions = model.predict(
                prediction_data
            )

            probabilities = model.predict_proba(
                prediction_data
            )[:, 1]

            # ----------------------------------------------
            # Results
            # ----------------------------------------------

            result = data.copy()

            result["Prediction"] = np.where(
                predictions == 1,
                "🚨 Fraudulent",
                "✅ Legitimate"
            )

            result["Fraud Probability (%)"] = (
                probabilities * 100
            ).round(2)

            # ----------------------------------------------
            # Statistics
            # ----------------------------------------------

            fraud_count = int(
                (predictions == 1).sum()
            )

            legitimate_count = int(
                (predictions == 0).sum()
            )

            total_count = len(predictions)

            fraud_percentage = (
                fraud_count / total_count
            ) * 100

            # ----------------------------------------------
            # Prediction Summary
            # ----------------------------------------------

            st.subheader("📊 Prediction Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🚨 Fraudulent",
                    f"{fraud_count:,}"
                )

            with col2:
                st.metric(
                    "✅ Legitimate",
                    f"{legitimate_count:,}"
                )

            with col3:
                st.metric(
                    "⚠️ Fraud Rate",
                    f"{fraud_percentage:.2f}%"
                )

            st.divider()

            # ----------------------------------------------
            # Distribution Chart
            # ----------------------------------------------

            st.subheader("📈 Transaction Distribution")

            chart_data = pd.DataFrame(
                {
                    "Transaction Type": [
                        "Legitimate",
                        "Fraudulent"
                    ],
                    "Count": [
                        legitimate_count,
                        fraud_count
                    ]
                }
            )

            st.bar_chart(
                chart_data.set_index(
                    "Transaction Type"
                )
            )

            st.divider()

            # ----------------------------------------------
            # Prediction Results
            # ----------------------------------------------

            st.subheader("🔎 Prediction Results")

            st.dataframe(
                result,
                use_container_width=True
            )

            # ----------------------------------------------
            # Download Results
            # ----------------------------------------------

            csv = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv,
                file_name="fraud_prediction_results.csv",
                mime="text/csv"
            )

            st.divider()

            st.success(
                f"✅ Analysis completed successfully for "
                f"{total_count:,} transactions."
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.warning(
                """
                Please make sure the uploaded CSV contains
                the same features used during model training.
                """
            )

            st.code(str(e))
