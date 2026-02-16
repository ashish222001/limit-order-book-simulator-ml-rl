Baseline model: 
# ml/baseline_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_baseline(
    data_path="ml/lob_labeled.csv",
    test_size=0.2,
    random_state=42
):
    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(data_path)

    # ---------------- FEATURES & LABEL ----------------
    feature_cols = [
        "best_bid",
        "best_ask",
        "spread",
        "bid_volume",
        "ask_volume",
        "imbalance",
        "midprice",
        "ltp",
    ]

    X = df[feature_cols]
    y = df["label"]

    # ---------------- TRAIN / TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ---------------- FEATURE SCALING ----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- MODEL ----------------
    model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
    )


    model.fit(X_train_scaled, y_train)

    # ---------------- EVALUATION ----------------
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    print("==== BASELINE LOGISTIC REGRESSION ====")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=====================================")


if __name__ == "__main__":
    run_baseline()
