import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from preprocessing import load_data, preprocess_data
from model import create_model


DATA_PATH = "creditcard.csv"
MODEL_PATH = "models/fraud_detection_model.pkl"


def main():

    # 1. Load dataset
    df = load_data(DATA_PATH)

    # 2. Preprocess data
    X, y = preprocess_data(df)

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Create model
    model = create_model()

    # 5. Train model
    model.fit(X_train, y_train)

    # 6. Save trained model
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print(f"Saved to: {MODEL_PATH}")

    # 7. Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 8. Evaluation
    print("\nModel Performance")
    print("=" * 40)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()