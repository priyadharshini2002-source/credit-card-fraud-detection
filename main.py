# ============================================================
# CREDIT CARD FRAUD DETECTION
# SMOTE + RANDOM FOREST
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import Counter

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df_credit = pd.read_csv("creditcard.csv")

print("=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df_credit.head())


print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(df_credit.info())


print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df_credit.describe())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df_credit.isnull().sum())

print("\nTotal missing values:",
      df_credit.isnull().sum().sum())


# ============================================================
# 4. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(df_credit["Class"].value_counts())

print("\nClass percentages:")

print(
    df_credit["Class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(4)
)


# ============================================================
# 5. CLASS COUNT VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df_credit,
    x="Class"
)

plt.title("Class Count", fontsize=18)
plt.xlabel("Class (0 = Normal, 1 = Fraud)", fontsize=13)
plt.ylabel("Count", fontsize=13)

plt.tight_layout()
plt.show()


# ============================================================
# 6. TIME FEATURE ENGINEERING
# ============================================================

# Original Time is measured in seconds.
# Convert it into hours and minutes.

timedelta = pd.to_timedelta(
    df_credit["Time"],
    unit="s"
)

df_credit["Time_hour"] = (
    timedelta.dt.components.hours
).astype(int)

df_credit["Time_min"] = (
    timedelta.dt.components.minutes
).astype(int)


print("\nTime features created:")
print(df_credit[["Time", "Time_hour", "Time_min"]].head())


# ============================================================
# 7. TRANSACTION DISTRIBUTION BY HOUR
# ============================================================

plt.figure(figsize=(12, 5))

sns.histplot(
    data=df_credit[df_credit["Class"] == 0],
    x="Time_hour",
    color="green",
    kde=True,
    stat="density",
    bins=24,
    alpha=0.4,
    label="Normal"
)

sns.histplot(
    data=df_credit[df_credit["Class"] == 1],
    x="Time_hour",
    color="red",
    kde=True,
    stat="density",
    bins=24,
    alpha=0.4,
    label="Fraud"
)

plt.title(
    "Fraud vs Normal Transactions by Hour",
    fontsize=17
)

plt.xlabel("Hour")
plt.ylabel("Density")
plt.xlim(-1, 24)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 8. TRANSACTION DISTRIBUTION BY MINUTE
# ============================================================

plt.figure(figsize=(12, 5))

sns.histplot(
    data=df_credit[df_credit["Class"] == 0],
    x="Time_min",
    color="green",
    kde=True,
    stat="density",
    bins=60,
    alpha=0.4,
    label="Normal"
)

sns.histplot(
    data=df_credit[df_credit["Class"] == 1],
    x="Time_min",
    color="red",
    kde=True,
    stat="density",
    bins=60,
    alpha=0.4,
    label="Fraud"
)

plt.title(
    "Fraud vs Normal Transactions by Minute",
    fontsize=17
)

plt.xlabel("Minute")
plt.ylabel("Density")
plt.xlim(-1, 60)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. SEPARATE FRAUD AND NORMAL TRANSACTIONS
# ============================================================

df_fraud = df_credit[
    df_credit["Class"] == 1
]

df_normal = df_credit[
    df_credit["Class"] == 0
]


# ============================================================
# 10. AMOUNT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FRAUD TRANSACTION AMOUNT STATISTICS")
print("=" * 70)

print(df_fraud["Amount"].describe())


print("\n" + "=" * 70)
print("NORMAL TRANSACTION AMOUNT STATISTICS")
print("=" * 70)

print(df_normal["Amount"].describe())


# ============================================================
# 11. LOG TRANSFORMATION OF AMOUNT
# ============================================================

# log1p(x) = log(1 + x)
# It handles Amount = 0 safely.

df_credit["Amount_log"] = np.log1p(
    df_credit["Amount"]
)


# ============================================================
# 12. AMOUNT BOX PLOTS
# ============================================================

plt.figure(figsize=(14, 6))


# Original Amount
plt.subplot(1, 2, 1)

sns.boxplot(
    data=df_credit,
    x="Class",
    y="Amount"
)

plt.title(
    "Class vs Transaction Amount",
    fontsize=18
)

plt.xlabel(
    "Class (0 = Normal, 1 = Fraud)"
)

plt.ylabel("Amount")


# Log Amount
plt.subplot(1, 2, 2)

sns.boxplot(
    data=df_credit,
    x="Class",
    y="Amount_log"
)

plt.title(
    "Class vs Log Transaction Amount",
    fontsize=18
)

plt.xlabel(
    "Class (0 = Normal, 1 = Fraud)"
)

plt.ylabel("Log Amount")


plt.tight_layout()
plt.show()


# ============================================================
# 13. AMOUNT VS TIME MINUTE
# ============================================================

plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=df_credit,
    x="Time_min",
    y="Amount",
    hue="Class",
    palette={
        0: "green",
        1: "red"
    },
    alpha=0.5
)

plt.title(
    "Transaction Amount by Minute",
    fontsize=16
)

plt.xlabel("Minute")
plt.ylabel("Amount")

plt.tight_layout()
plt.show()


# ============================================================
# 14. AMOUNT VS TIME HOUR
# ============================================================

plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=df_credit,
    x="Time_hour",
    y="Amount",
    hue="Class",
    palette={
        0: "green",
        1: "red"
    },
    alpha=0.5
)

plt.title(
    "Transaction Amount by Hour",
    fontsize=16
)

plt.xlabel("Hour")
plt.ylabel("Amount")

plt.tight_layout()
plt.show()


# ============================================================
# 15. DISTRIBUTION OF V FEATURES
# ============================================================

columns = df_credit.iloc[:, 1:29].columns

frauds = df_credit["Class"] == 1
normals = df_credit["Class"] == 0


plt.figure(
    figsize=(15, 70)
)

grid = gridspec.GridSpec(
    14,
    2
)


for n, col in enumerate(columns):

    ax = plt.subplot(grid[n])

    sns.kdeplot(
        df_credit.loc[frauds, col],
        color="red",
        fill=True,
        alpha=0.3,
        label="Fraud"
    )

    sns.kdeplot(
        df_credit.loc[normals, col],
        color="green",
        fill=True,
        alpha=0.3,
        label="Normal"
    )

    ax.set_ylabel("Density")
    ax.set_title(
        str(col),
        fontsize=12
    )

    ax.set_xlabel("")

    if n == 0:
        ax.legend()


plt.tight_layout()
plt.show()


# ============================================================
# 16. SELECT FEATURES
# ============================================================

features = [
    "Time_hour",
    "Time_min",
    "V2",
    "V3",
    "V4",
    "V9",
    "V10",
    "V11",
    "V12",
    "V14",
    "V16",
    "V17",
    "V18",
    "V19",
    "V27",
    "Amount"
]


df_credit = df_credit[
    features + ["Class"]
].copy()


# ============================================================
# 17. LOG TRANSFORM AMOUNT
# ============================================================

df_credit["Amount"] = np.log1p(
    df_credit["Amount"]
)


print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(df_credit.head())

print("\nShape:", df_credit.shape)


# ============================================================
# 18. CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(14, 12)
)

colormap = plt.cm.Greens

sns.heatmap(
    df_credit.corr(),
    linewidths=0.1,
    vmax=1.0,
    vmin=-1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
    fmt=".2f"
)

plt.title(
    "Feature Correlation Heatmap",
    fontsize=18
)

plt.tight_layout()
plt.show()


# ============================================================
# 19. CREATE X AND Y
# ============================================================

X = df_credit.drop(
    "Class",
    axis=1
)

y = df_credit["Class"]


print("\n" + "=" * 70)
print("ORIGINAL DATA DISTRIBUTION")
print("=" * 70)

print(Counter(y))


# ============================================================
# 20. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set:", X_train.shape)
print("Testing set :", X_test.shape)


print("\nTraining class distribution:")
print(Counter(y_train))


print("\nTesting class distribution:")
print(Counter(y_test))


# ============================================================
# 21. CREATE EVALUATION FUNCTION
# ============================================================

def print_results(
    model_name,
    true_value,
    pred,
    pred_prob=None
):

    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        "Accuracy :",
        round(
            accuracy_score(
                true_value,
                pred
            ),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                true_value,
                pred,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall   :",
        round(
            recall_score(
                true_value,
                pred,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1 Score :",
        round(
            f1_score(
                true_value,
                pred,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F2 Score :",
        round(
            fbeta_score(
                true_value,
                pred,
                beta=2,
                zero_division=0
            ),
            4
        )
    )

    if pred_prob is not None:

        print(
            "PR-AUC   :",
            round(
                average_precision_score(
                    true_value,
                    pred_prob
                ),
                4
            )
        )

        print(
            "ROC-AUC  :",
            round(
                roc_auc_score(
                    true_value,
                    pred_prob
                ),
                4
            )
        )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            true_value,
            pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            true_value,
            pred,
            zero_division=0
        )
    )


# ============================================================
# 22. SMOTE + RANDOM FOREST PIPELINE
# ============================================================

smote_rf_pipeline = Pipeline(
    steps=[
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 23. TRAIN SMOTE + RANDOM FOREST
# ============================================================

print("\nTraining SMOTE + Random Forest...")

smote_rf_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 24. PREDICTIONS
# ============================================================

y_pred_smote = (
    smote_rf_pipeline.predict(
        X_test
    )
)


y_pred_prob_smote = (
    smote_rf_pipeline.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 25. EVALUATE SMOTE + RANDOM FOREST
# ============================================================

print_results(
    "SMOTE + Random Forest",
    y_test,
    y_pred_smote,
    y_pred_prob_smote
)


# ============================================================
# 26. CHECK SMOTE DISTRIBUTION
# ============================================================

X_train_smote, y_train_smote = SMOTE(
    random_state=42
).fit_resample(
    X_train,
    y_train
)


print("\n" + "=" * 70)
print("CLASS DISTRIBUTION AFTER SMOTE")
print("=" * 70)

print(
    Counter(y_train_smote)
)


# ============================================================
# 27. PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds = (
    precision_recall_curve(
        y_test,
        y_pred_prob_smote
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision,
    color="blue",
    linewidth=2
)

plt.xlabel(
    "Recall",
    fontsize=13
)

plt.ylabel(
    "Precision",
    fontsize=13
)

plt.title(
    "Precision-Recall Curve",
    fontsize=16
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# 28. ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = (
    roc_curve(
        y_test,
        y_pred_prob_smote
    )
)


roc_auc = roc_auc_score(
    y_test,
    y_pred_prob_smote
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    color="darkorange",
    linewidth=2,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# 29. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)


pipeline = Pipeline(
    steps=[
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "rf",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {

    "rf__max_depth": [
        3,
        5,
        10,
        None
    ],

    "rf__n_estimators": [
        50,
        100,
        200
    ],

    "rf__max_features": [
        "sqrt",
        "log2"
    ],

    "rf__min_samples_leaf": [
        1,
        2,
        5
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="recall",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 30. BEST PARAMETERS
# ============================================================

print("\nBest CV Recall:")

print(
    grid_search.best_score_
)


print("\nBest Parameters:")

print(
    grid_search.best_params_
)


# ============================================================
# 31. BEST MODEL
# ============================================================

best_model = (
    grid_search.best_estimator_
)


# ============================================================
# 32. TEST BEST MODEL
# ============================================================

y_pred = (
    best_model.predict(
        X_test
    )
)


y_pred_prob = (
    best_model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 33. FINAL MODEL EVALUATION
# ============================================================

print_results(
    "Optimized SMOTE + Random Forest",
    y_test,
    y_pred,
    y_pred_prob
)


# ============================================================
# 34. CONFUSION MATRIX VISUALIZATION
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Normal",
        "Fraud"
    ],
    yticklabels=[
        "Normal",
        "Fraud"
    ]
)

plt.title(
    "Confusion Matrix",
    fontsize=16
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()
plt.show()


# ============================================================
# 35. FINAL PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds = (
    precision_recall_curve(
        y_test,
        y_pred_prob
    )
)


pr_auc = average_precision_score(
    y_test,
    y_pred_prob
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision,
    color="blue",
    linewidth=2,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Optimized Random Forest"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# 36. FINAL ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_pred_prob
)

roc_auc = roc_auc_score(
    y_test,
    y_pred_prob
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    color="darkorange",
    linewidth=2,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Optimized Random Forest"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# 37. FEATURE IMPORTANCE
# ============================================================

rf_model = (
    best_model
    .named_steps["rf"]
)


feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        rf_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance
)


# ============================================================
# 38. FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(
    figsize=(10, 7)
)

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
    color="seagreen"
)

plt.title(
    "Feature Importance - Random Forest",
    fontsize=18
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()
plt.show()


# ============================================================
# 39. CROSS-VALIDATION RECALL
# ============================================================

print("\n" + "=" * 70)
print("10-FOLD CROSS-VALIDATION")
print("=" * 70)


cv_10 = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


cv_results = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=cv_10,
    scoring="recall",
    n_jobs=-1
)


print(
    "Recall for each fold:"
)

print(
    cv_results
)


print(
    "\nMean Recall:",
    round(
        cv_results.mean(),
        4
    )
)


print(
    "Standard Deviation:",
    round(
        cv_results.std(),
        4
    )
)


# ============================================================
# 40. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    "Accuracy :",
    round(
        accuracy_score(
            y_test,
            y_pred
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall   :",
    round(
        recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "F1 Score :",
    round(
        f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "F2 Score :",
    round(
        fbeta_score(
            y_test,
            y_pred,
            beta=2,
            zero_division=0
        ),
        4
    )
)

print(
    "PR-AUC   :",
    round(
        average_precision_score(
            y_test,
            y_pred_prob
        ),
        4
    )
)

print(
    "ROC-AUC  :",
    round(
        roc_auc_score(
            y_test,
            y_pred_prob
        ),
        4
    )
)

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)
