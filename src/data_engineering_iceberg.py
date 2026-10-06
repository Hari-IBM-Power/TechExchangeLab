"""
Data Engineering Module for Time Series Analysis — Iceberg Edition
Handles data loading, preprocessing, and feature engineering for user savings prediction.
Reads data from watsonx.data Iceberg tables via Presto/SQLAlchemy instead of CSV files.

Connection details are read from src/iceberg.json.
Credentials are read from environment variables LH_USER and LH_PW (via .env).

Tables used:
    iceberg_data.pfa.salary_transactions_2025
    iceberg_data.pfa.users
    iceberg_data.pfa.retirement
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, cast

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Bootstrap: load .env, parse iceberg.json, write SSL cert to disk
# ---------------------------------------------------------------------------

load_dotenv()

_SRC_DIR = Path(__file__).parent
_ICEBERG_JSON = _SRC_DIR / "iceberg.json"
_CERT_PATH = _SRC_DIR / "cert.crt"

_LH_CATALOG = "iceberg_data"
_LH_SCHEMA = "pfa"


def _parse_iceberg_json(json_path: Path) -> dict:
    """Parse src/iceberg.json and return a flat dict keyed by connection property name."""
    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {item["name"]: item["value"] for item in raw["properties"]["connection"]}


_conn_props = _parse_iceberg_json(_ICEBERG_JSON)

# Write the SSL certificate to disk so requests can use it as a verify path
_CERT_PATH.write_text(_conn_props["ssl_certificate"], encoding="utf-8")

_LH_HOST = _conn_props["engine_host"]
_LH_PORT = _conn_props["engine_port"]


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _get_engine() -> Engine:
    """
    Build a SQLAlchemy Presto engine connected to iceberg_data/pfa.

    Credentials are read from env vars LH_USER and LH_PW.

    Raises:
        EnvironmentError: if LH_USER or LH_PW are not set.
    """
    lh_user = os.getenv("LH_USER")
    lh_pw = os.getenv("LH_PW")

    if not lh_user:
        raise EnvironmentError("LH_USER environment variable is not set.")
    if not lh_pw:
        raise EnvironmentError("LH_PW environment variable is not set.")

    engine = create_engine(
        f"presto://{lh_user}:{lh_pw}@{_LH_HOST}:{_LH_PORT}/{_LH_CATALOG}/{_LH_SCHEMA}",
        connect_args={
            "protocol": "https",
            "requests_kwargs": {"verify": str(_CERT_PATH)},
        },
    )
    return engine


# ---------------------------------------------------------------------------
# Data loading functions (Iceberg replacements for CSV reads)
# ---------------------------------------------------------------------------

def load_transaction_data(filepath: str = None) -> pd.DataFrame:  # type: ignore[assignment]
    """
    Load transaction data from the Iceberg table salary_transactions_2025.

    Parameters:
    -----------
    filepath : str, optional
        Ignored. Kept for API compatibility with data_engineering.py.

    Returns:
    --------
    pd.DataFrame
        Loaded transaction data.
    """
    engine = _get_engine()
    df = pd.read_sql("SELECT * FROM salary_transactions_2025", engine)
    # Presto returns lowercase column names; normalise to Title Case so the
    # rest of the pipeline (preprocess_data, calculate_monthly_savings, etc.)
    # can reference 'Year', 'Month', 'Day', 'User', 'Amount', 'Balance', 'Time'.
    df.columns = [c.title() for c in df.columns]
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess transaction data.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data.

    Returns:
    --------
    pd.DataFrame
        Preprocessed data with datetime column.
    """
    # Create datetime column
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Sort by user and date
    df = df.sort_values(['User', 'Date', 'Time']).reset_index(drop=True)

    return df


def calculate_monthly_savings(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Calculate monthly savings for a specific user.
    Savings = End of month balance - Start of month balance.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed transaction data.
    user_id : int
        User ID to filter data.

    Returns:
    --------
    pd.DataFrame
        Monthly savings data with columns: Year, Month, Savings, EndBalance.
    """
    user_df = df[df['User'] == user_id].copy()

    if user_df.empty:
        raise ValueError(f"No data found for user {user_id}")

    monthly_data = []

    for key, group in user_df.groupby(['Year', 'Month']):
        year, month = key  # type: ignore[misc]
        start_balance = group.iloc[0]['Balance'] - group.iloc[0]['Amount']
        end_balance = group.iloc[-1]['Balance']
        savings = end_balance - start_balance

        monthly_data.append({
            'Year': year,
            'Month': month,
            'Savings': savings,
            'EndBalance': end_balance,
            'StartBalance': start_balance
        })

    monthly_df = pd.DataFrame(monthly_data)

    monthly_df['Date'] = pd.to_datetime(monthly_df[['Year', 'Month']].assign(Day=1))
    monthly_df = monthly_df.sort_values('Date').reset_index(drop=True)

    return monthly_df


def prepare_timeseries_data(monthly_df: pd.DataFrame) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    Prepare time series data for modeling.

    Parameters:
    -----------
    monthly_df : pd.DataFrame
        Monthly savings data.

    Returns:
    --------
    Tuple[pd.Series, pd.DatetimeIndex]
        Time series of savings and datetime index.
    """
    ts_data = monthly_df.set_index('Date')['Savings']

    ts_data.index = pd.DatetimeIndex(ts_data.index)
    ts_data = cast(pd.Series, ts_data.asfreq('MS'))

    return ts_data, cast(pd.DatetimeIndex, ts_data.index)


def get_user_savings_timeseries(
    filepath: str = None,  # type: ignore[assignment]
    user_id: int = 0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Complete pipeline to get user savings time series from Iceberg.

    Parameters:
    -----------
    filepath : str, optional
        Ignored. Kept for API compatibility with data_engineering.py.
    user_id : int
        User ID to process.

    Returns:
    --------
    Tuple[pd.Series, pd.DataFrame]
        Time series data and monthly dataframe with additional info.
    """
    df = load_transaction_data()
    df = preprocess_data(df)

    monthly_df = calculate_monthly_savings(df, user_id)

    ts_data, _ = prepare_timeseries_data(monthly_df)

    return ts_data, monthly_df


def validate_user_exists(filepath: str = None, user_id: int = 0) -> bool:  # type: ignore[assignment]
    """
    Check if user exists in the Iceberg transactions table.

    Parameters:
    -----------
    filepath : str, optional
        Ignored. Kept for API compatibility with data_engineering.py.
    user_id : int
        User ID to check.

    Returns:
    --------
    bool
        True if user exists, False otherwise.
    """
    engine = _get_engine()
    query = text('SELECT DISTINCT "User" FROM salary_transactions_2025')
    with engine.connect() as conn:
        result = conn.execute(query)
        users = [row[0] for row in result]
    return user_id in users


def get_available_users(filepath: str = None) -> list:  # type: ignore[assignment]
    """
    Get list of available user IDs from the Iceberg transactions table.

    Parameters:
    -----------
    filepath : str, optional
        Ignored. Kept for API compatibility with data_engineering.py.

    Returns:
    --------
    list
        Sorted list of available user IDs.
    """
    engine = _get_engine()
    query = text('SELECT DISTINCT "User" FROM salary_transactions_2025')
    with engine.connect() as conn:
        result = conn.execute(query)
        users = [row[0] for row in result]
    return sorted(users)


def get_user_details(filepath: str = None, user_id: int = 0) -> Dict:  # type: ignore[assignment]
    """
    Get user profile details from the Iceberg users table.

    Parameters:
    -----------
    filepath : str, optional
        Ignored. Kept for API compatibility with data_engineering.py.
    user_id : int
        User ID to look up.

    Returns:
    --------
    Dict
        User details as a dictionary.
    """
    engine = _get_engine()
    query = text('SELECT * FROM users WHERE "User ID" = :uid')
    with engine.connect() as conn:
        result = conn.execute(query, {"uid": user_id})
        rows = result.fetchall()
        columns = list(result.keys())

    if not rows:
        raise ValueError(f"No user details found for user {user_id}")

    user_details = dict(zip(columns, rows[0]))

    for key, value in user_details.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            user_details[key] = None

    return user_details

# Made with Bob


if __name__ == "__main__":
    _engine = _get_engine()
    _tables = [
        "salary_transactions_2025",
        "users",
        "retirement",
    ]
    for _table in _tables:
        print(f"\n{'='*60}")
        print(f"Table: {_LH_CATALOG}.{_LH_SCHEMA}.{_table}")
        print('='*60)
        _df = pd.read_sql(f"SELECT * FROM {_table} LIMIT 1", _engine)
        print(_df.to_string(index=False))
