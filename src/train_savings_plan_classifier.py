"""Train and save a savings plan classifier using logistic regression."""
import pickle
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "../new/synthetic_retirement_data.csv"
MODEL_PATH = "./models/savings_plan_classifier.pkl"

CANONICAL_COLUMNS = [
    "Current Age",
    "Retirement Age",
    "Birth Year",
    "Birth Month",
    "Gender",
    "City",
    "State",
    "Zipcode",
    "Per Capita Income - Zipcode",
    "Yearly Income - Person",
    "Total Debt",
    "FICO Score",
    "Savings Plan",
]

NUMERIC_COLUMNS = [
    "Current Age",
    "Retirement Age",
    "Birth Year",
    "Birth Month",
    "Per Capita Income - Zipcode",
    "Yearly Income - Person",
    "Total Debt",
    "FICO Score",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "City",
    "State",
    "Zipcode",
]

COLUMN_ALIASES = {
    "Current Age": {
        "current age",
    },
    "Retirement Age": {
        "retirement age",
    },
    "Birth Year": {
        "birth year",
    },
    "Birth Month": {
        "birth month",
    },
    "Gender": {
        "gender",
    },
    "City": {
        "city",
    },
    "State": {
        "state",
    },
    "Zipcode": {
        "zipcode",
        "zip code",
        "zip",
    },
    "Per Capita Income - Zipcode": {
        "per capita income - zipcode",
        "per capita income zipcode",
        "per capita income zip code",
        "per capita income - zip code",
        "per capita income – zipcode",
        "per capita income – zip code",
    },
    "Yearly Income - Person": {
        "yearly income - person",
        "yearly income person",
        "annual income - person",
        "annual income person",
        "yearly income – person",
    },
    "Total Debt": {
        "total debt",
    },
    "FICO Score": {
        "fico score",
    },
    "Savings Plan": {
        "savings plan",
    },
}


def normalize_text(value):
    """Normalize whitespace and dashes in a string value."""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace("–", "-")
    return value


def canonical_key(value):
    """Return a canonical (lowercase, normalized) key for a column name."""
    return normalize_text(value).lower()


def standardize_columns(df) -> pd.DataFrame:
    """Rename dataframe columns to canonical names using COLUMN_ALIASES."""
    df = df.copy()
    df.columns = [normalize_text(col) for col in df.columns]

    rename_map = {}
    for column in df.columns:
        key = canonical_key(column)
        for canonical_name, aliases in COLUMN_ALIASES.items():
            if key in aliases:
                rename_map[column] = canonical_name
                break

    df = df.rename(columns=rename_map)

    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    return df[CANONICAL_COLUMNS].copy()


def clean_dataframe(df) -> pd.DataFrame:
    """Clean and validate the dataframe, coercing invalid values to NA."""
    df = df.copy()

    df["Savings Plan"] = df["Savings Plan"].astype(str).str.strip()
    df = df[df["Savings Plan"].ne("")]
    df = df[df["Savings Plan"].str.lower().ne("nan")]

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[~df["Birth Month"].between(1, 12), "Birth Month"] = pd.NA
    df.loc[~df["FICO Score"].between(300, 850), "FICO Score"] = pd.NA
    df.loc[~df["Current Age"].between(18, 100), "Current Age"] = pd.NA
    df.loc[~df["Retirement Age"].between(18, 100), "Retirement Age"] = pd.NA
    df.loc[~df["Birth Year"].between(1900, 2100), "Birth Year"] = pd.NA

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA

    return df


def build_pipeline():
    """Build and return the preprocessing + logistic regression pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    """Load data, train the pipeline, evaluate, and save the model."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    df = standardize_columns(df)
    print(f"Columns after normalization: {list(df.columns)}")

    df = clean_dataframe(df)
    print(f"Rows after cleaning target blanks: {len(df)}")
    print("Missing values by column:")
    print(df.isna().sum().to_string())

    features = df.drop(columns=["Savings Plan"])
    y = df["Savings Plan"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    print("\nEvaluation Summary")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(pipeline, file)

    print(f"\nSaved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()

# Made with Bob
