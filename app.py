import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "models/fraud_detection_model.pkl"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("💳 Fraud Detection")

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

    st.title("💳 Credit Card Fraud Detection")

    st.write(
        "Machine Learning based system for detecting "
        "potentially fraudulent credit card transactions."
    )

    st.divider()

    st.subheader("📌 Project Overview")

    st.write(
        "This application uses a Random Forest Machine Learning "
        "model with SMOTE to identify suspicious credit card transactions."
    )

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "⚖️ Imbalance Technique",
            "SMOTE"
        )

    with col3:
        st.metric(
            "📊 ROC-AUC",
            "0.9689"
        )

    st.divider()

    st.subheader("🛠️ Machine Learning Pipeline")

    st.write(
        "Data Loading → Data Preprocessing → "
        "Class Imbalance Handling → Random Forest → "
        "Fraud Prediction"
    )

    st.divider()

    st.subheader("🎯 Key Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.write("✅ Fraudulent transaction detection")
        st.write("✅ SMOTE for imbalanced data")
        st.write("✅ Random Forest classification")

    with feature_col2:
        st.write("✅ Fraud probability")
        st.write("✅ Transaction distribution")
        st.write("✅ Prediction result download")

    st.info(
        "🚀 Go to **🔍 Fraud Prediction** from the sidebar "
        "to analyse transactions."
    )


# --------------------------------------------------
# FRAUD PREDICTION PAGE
# --------------------------------------------------

elif page == "🔍 Fraud Prediction":

    st.title("🔍 Fraud Transaction Prediction")

    st.write(
        "Upload a credit card transaction CSV file "
        "to detect potentially fraudulent transactions."
    )

    st.divider()

    # Upload CSV
    uploaded_file = st.file_uploader(
        "📂 Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Read data
        data = pd.read_csv(uploaded_file)

        st.success("✅ File uploaded successfully!")

        # --------------------------------------------------
        # DATA PREVIEW
        # --------------------------------------------------

        st.subheader("📄 Uploaded Data")

        st.write(
            f"Total Transactions: **{len(data):,}**"
        )

        st.dataframe(
            data.head(),
            use_container_width=True
        )

        # --------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------

        prediction_data = data.copy()

        # Remove target column if dataset contains it
        if "Class" in prediction_data.columns:
            prediction_data = prediction_data.drop(
                "Class",
                axis=1
            )

        # Same transformation used during training
        if "Amount" in prediction_data.columns:
            prediction_data["Amount"] = np.log1p(
                prediction_data["Amount"]
            )

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        try:

            predictions = model.predict(
                prediction_data
            )

            probabilities = model.predict_proba(
                prediction_data
            )[:, 1]

            # --------------------------------------------------
            # RESULTS
            # --------------------------------------------------

            result = data.copy()

            result["Prediction"] = np.where(
                predictions == 1,
                "🚨 Fraudulent",
                "✅ Legitimate"
            )

            result["Fraud Probability"] = (
                probabilities * 100
            ).round(2)

            # --------------------------------------------------
            # COUNTS
            # --------------------------------------------------

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

            # --------------------------------------------------
            # SUMMARY
            # --------------------------------------------------

            st.divider()

            st.subheader("📊 Prediction Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🚨 Fraudulent Transactions",
                    f"{fraud_count:,}"
                )

            with col2:
                st.metric(
                    "✅ Legitimate Transactions",
                    f"{legitimate_count:,}"
                )

            with col3:
                st.metric(
                    "⚠️ Fraud Rate",
                    f"{fraud_percentage:.2f}%"
                )

            # --------------------------------------------------
            # CHART
            # --------------------------------------------------

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

            # --------------------------------------------------
            # PREDICTION RESULTS
            # --------------------------------------------------

            st.subheader("🔎 Prediction Results")

            st.dataframe(
                result,
                use_container_width=True
            )

            # --------------------------------------------------
            # DOWNLOAD RESULTS
            # --------------------------------------------------

            csv = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv,
                file_name="fraud_prediction_results.csv",
                mime="text/csv"
            )

            # --------------------------------------------------
            # SUCCESS MESSAGE
            # --------------------------------------------------

            st.success(
                f"✅ Analysis completed for "
                f"{total_count:,} transactions."
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.write(
                "Please make sure the uploaded CSV "
                "contains the same features used during training."
            )

            st.code(str(e))