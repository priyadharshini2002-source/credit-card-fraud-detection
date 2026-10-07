import joblib
import pandas as pd


MODEL_PATH = "models/fraud_detection_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_transaction(transaction):
    model = load_model()

    # Convert transaction data into DataFrame
    transaction_df = pd.DataFrame([transaction])

    # Apply same preprocessing used during training
    transaction_df["Amount"] = transaction_df["Amount"].apply(
        lambda x: __import__("numpy").log1p(x)
    )

    prediction = model.predict(transaction_df)[0]
    probability = model.predict_proba(transaction_df)[0][1]

    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return {
        "prediction": result,
        "fraud_probability": probability
    }


if __name__ == "__main__":
    print("Credit Card Fraud Detection")
    print("=" * 40)
    print("Prediction module loaded successfully.")