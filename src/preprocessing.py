import pandas as pd
import numpy as np


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    df = df.copy()

    # Log transformation for transaction amount
    df["Amount"] = np.log1p(df["Amount"])

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y