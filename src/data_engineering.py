"""
Data Engineering Module for Time Series Analysis
Handles data loading, preprocessing, and feature engineering for user savings prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


def load_transaction_data(filepath: str) -> pd.DataFrame:
    """
    Load transaction data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing transaction data
        
    Returns:
    --------
    pd.DataFrame
        Loaded transaction data
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess transaction data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data with datetime column
    """
    # Create datetime column
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    
    # Sort by user and date
    df = df.sort_values(['User', 'Date', 'Time']).reset_index(drop=True)
    
    return df


def calculate_monthly_savings(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Calculate monthly savings for a specific user
    Savings = End of month balance - Start of month balance
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed transaction data
    user_id : int
        User ID to filter data
        
    Returns:
    --------
    pd.DataFrame
        Monthly savings data with columns: Year, Month, Savings, EndBalance
    """
    # Filter data for specific user
    user_df = df[df['User'] == user_id].copy()
    
    if user_df.empty:
        raise ValueError(f"No data found for user {user_id}")
    
    # Group by year and month
    monthly_data = []
    
    for (year, month), group in user_df.groupby(['Year', 'Month']):
        # Get first and last balance of the month
        start_balance = group.iloc[0]['Balance'] - group.iloc[0]['Amount']
        end_balance = group.iloc[-1]['Balance']
        
        # Calculate savings (positive means saved money, negative means spent more)
        savings = end_balance - start_balance
        
        monthly_data.append({
            'Year': year,
            'Month': month,
            'Savings': savings,
            'EndBalance': end_balance,
            'StartBalance': start_balance
        })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    # Create a proper date column for time series
    monthly_df['Date'] = pd.to_datetime(monthly_df[['Year', 'Month']].assign(Day=1))
    monthly_df = monthly_df.sort_values('Date').reset_index(drop=True)
    
    return monthly_df


def prepare_timeseries_data(monthly_df: pd.DataFrame) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    Prepare time series data for modeling
    
    Parameters:
    -----------
    monthly_df : pd.DataFrame
        Monthly savings data
        
    Returns:
    --------
    Tuple[pd.Series, pd.DatetimeIndex]
        Time series of savings and datetime index
    """
    # Set date as index
    ts_data = monthly_df.set_index('Date')['Savings']
    
    # Ensure the index is a proper datetime index with monthly frequency
    ts_data.index = pd.DatetimeIndex(ts_data.index)
    ts_data = ts_data.asfreq('MS')  # Month Start frequency
    
    return ts_data, ts_data.index


def get_user_savings_timeseries(filepath: str, user_id: int) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Complete pipeline to get user savings time series
    
    Parameters:
    -----------
    filepath : str
        Path to transaction data CSV
    user_id : int
        User ID to process
        
    Returns:
    --------
    Tuple[pd.Series, pd.DataFrame]
        Time series data and monthly dataframe with additional info
    """
    # Load and preprocess data
    df = load_transaction_data(filepath)
    df = preprocess_data(df)
    
    # Calculate monthly savings
    monthly_df = calculate_monthly_savings(df, user_id)
    
    # Prepare time series
    ts_data, _ = prepare_timeseries_data(monthly_df)
    
    return ts_data, monthly_df


def validate_user_exists(filepath: str, user_id: int) -> bool:
    """
    Check if user exists in the dataset
    
    Parameters:
    -----------
    filepath : str
        Path to transaction data CSV
    user_id : int
        User ID to check
        
    Returns:
    --------
    bool
        True if user exists, False otherwise
    """
    df = pd.read_csv(filepath)
    return user_id in df['User'].unique()


def get_available_users(filepath: str) -> list:
    """
    Get list of available user IDs in the dataset
    
    Parameters:
    -----------
    filepath : str
        Path to transaction data CSV
        
    Returns:
    --------
    list
        List of available user IDs
    """
    df = pd.read_csv(filepath)
    return sorted(df['User'].unique().tolist())


def get_user_details(filepath: str, user_id: int) -> Dict:
    """
    Get user profile details for a specific user from the users CSV.

    Parameters:
    -----------
    filepath : str
        Path to user details CSV
    user_id : int
        User ID to look up

    Returns:
    --------
    Dict
        User details as a dictionary
    """
    df = pd.read_csv(filepath)
    user_row = df[df['User ID'] == user_id]

    if user_row.empty:
        raise ValueError(f"No user details found for user {user_id}")

    user_details = user_row.iloc[0].to_dict()

    for key, value in user_details.items():
        if pd.isna(value):
            user_details[key] = None

    return user_details

# Made with Bob
