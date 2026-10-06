"""
Unit tests for data_engineering module
"""
import pytest
import pandas as pd
import numpy as np
from src.data_engineering_iceberg import (
    load_transaction_data,
    preprocess_data,
    calculate_monthly_savings,
    prepare_timeseries_data,
    get_user_savings_timeseries,
    validate_user_exists,
    get_available_users,
    get_user_details
)


class TestLoadTransactionData:
    """Tests for load_transaction_data function"""

    def test_load_valid_csv(self, sample_csv_file):
        """Test loading a valid CSV file"""
        df = load_transaction_data(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'User' in df.columns
        assert 'Balance' in df.columns

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_transaction_data('nonexistent_file.csv')


class TestPreprocessData:
    """Tests for preprocess_data function"""

    def test_preprocess_creates_date_column(self, sample_transaction_data):
        """Test that preprocessing creates a Date column"""
        df = preprocess_data(sample_transaction_data)
        assert 'Date' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['Date'])

    def test_preprocess_sorts_data(self, sample_transaction_data):
        """Test that data is sorted by User, Date, and Time"""
        # Shuffle the data
        shuffled = sample_transaction_data.sample(frac=1).reset_index(drop=True)
        df = preprocess_data(shuffled)

        # Check if sorted
        assert df['User'].is_monotonic_increasing or not df['User'].is_monotonic_decreasing

    def test_preprocess_preserves_all_rows(self, sample_transaction_data):
        """Test that no rows are lost during preprocessing"""
        original_len = len(sample_transaction_data)
        df = preprocess_data(sample_transaction_data)
        assert len(df) == original_len


class TestCalculateMonthlySavings:
    """Tests for calculate_monthly_savings function"""

    def test_calculate_savings_valid_user(self, sample_csv_file):
        """Test calculating savings for a valid user"""
        df = load_transaction_data(sample_csv_file)
        df = preprocess_data(df)
        monthly_df = calculate_monthly_savings(df, user_id=0)

        assert isinstance(monthly_df, pd.DataFrame)
        assert not monthly_df.empty
        assert 'Savings' in monthly_df.columns
        assert 'EndBalance' in monthly_df.columns
        assert 'StartBalance' in monthly_df.columns
        assert 'Date' in monthly_df.columns

    def test_calculate_savings_invalid_user(self, sample_csv_file):
        """Test that invalid user raises ValueError"""
        df = load_transaction_data(sample_csv_file)
        df = preprocess_data(df)

        with pytest.raises(ValueError, match="No data found for user"):
            calculate_monthly_savings(df, user_id=999)

    def test_savings_calculation_logic(self, sample_csv_file):
        """Test that savings are calculated correctly"""
        df = load_transaction_data(sample_csv_file)
        df = preprocess_data(df)
        monthly_df = calculate_monthly_savings(df, user_id=0)

        # Savings should be EndBalance - StartBalance
        for _, row in monthly_df.iterrows():
            expected_savings = row['EndBalance'] - row['StartBalance']
            assert abs(row['Savings'] - expected_savings) < 0.01

    def test_monthly_df_has_date_column(self, sample_csv_file):
        """Test that monthly dataframe has proper Date column"""
        df = load_transaction_data(sample_csv_file)
        df = preprocess_data(df)
        monthly_df = calculate_monthly_savings(df, user_id=0)

        assert 'Date' in monthly_df.columns
        assert pd.api.types.is_datetime64_any_dtype(monthly_df['Date'])

    def test_monthly_df_sorted_by_date(self, sample_csv_file):
        """Test that monthly dataframe is sorted by date"""
        df = load_transaction_data(sample_csv_file)
        df = preprocess_data(df)
        monthly_df = calculate_monthly_savings(df, user_id=0)

        dates = monthly_df['Date'].tolist()
        assert dates == sorted(dates)


class TestPrepareTimeseriesData:
    """Tests for prepare_timeseries_data function"""

    def test_prepare_timeseries_returns_series(self, sample_monthly_df):
        """Test that function returns a pandas Series"""
        ts_data, ts_index = prepare_timeseries_data(sample_monthly_df)
        assert isinstance(ts_data, pd.Series)
        assert isinstance(ts_index, pd.DatetimeIndex)

    def test_timeseries_has_datetime_index(self, sample_monthly_df):
        """Test that time series has datetime index"""
        ts_data, _ = prepare_timeseries_data(sample_monthly_df)
        assert isinstance(ts_data.index, pd.DatetimeIndex)

    def test_timeseries_frequency(self, sample_monthly_df):
        """Test that time series has monthly frequency"""
        _, ts_index = prepare_timeseries_data(sample_monthly_df)
        assert isinstance(ts_index, pd.DatetimeIndex)
        assert str(ts_index.freq) == 'MS' or ts_index.inferred_freq == 'MS'

    def test_timeseries_values_match(self, sample_monthly_df):
        """Test that time series values match the Savings column"""
        ts_data, _ = prepare_timeseries_data(sample_monthly_df)
        expected_values = sample_monthly_df['Savings'].astype(float).to_numpy()
        actual_values = ts_data.astype(float).to_numpy()
        np.testing.assert_array_almost_equal(actual_values, expected_values)


class TestGetUserSavingsTimeseries:
    """Tests for get_user_savings_timeseries function"""

    def test_get_timeseries_valid_user(self, sample_csv_file):
        """Test getting time series for a valid user"""
        ts_data, monthly_df = get_user_savings_timeseries(sample_csv_file, user_id=0)

        assert isinstance(ts_data, pd.Series)
        assert isinstance(monthly_df, pd.DataFrame)
        assert not ts_data.empty
        assert not monthly_df.empty

    def test_get_timeseries_invalid_user(self, sample_csv_file):
        """Test that invalid user raises ValueError"""
        with pytest.raises(ValueError):
            get_user_savings_timeseries(sample_csv_file, user_id=999)

    def test_timeseries_and_df_lengths_match(self, sample_csv_file):
        """Test that time series and dataframe have same length"""
        ts_data, monthly_df = get_user_savings_timeseries(sample_csv_file, user_id=0)
        assert len(ts_data) == len(monthly_df)


class TestValidateUserExists:
    """Tests for validate_user_exists function"""

    def test_validate_existing_user(self, sample_csv_file):
        """Test validation for existing user"""
        assert validate_user_exists(sample_csv_file, user_id=0) is True
        assert validate_user_exists(sample_csv_file, user_id=1) is True

    def test_validate_nonexistent_user(self, sample_csv_file):
        """Test validation for non-existent user"""
        assert validate_user_exists(sample_csv_file, user_id=999) is False
        assert validate_user_exists(sample_csv_file, user_id=-1) is False


class TestGetAvailableUsers:
    """Tests for get_available_users function"""

    def test_get_available_users_returns_list(self, sample_csv_file):
        """Test that function returns a list"""
        users = get_available_users(sample_csv_file)
        assert isinstance(users, list)

    def test_get_available_users_correct_users(self, sample_csv_file):
        """Test that correct users are returned"""
        users = get_available_users(sample_csv_file)
        assert 0 in users
        assert 1 in users
        assert len(users) == 2

    def test_get_available_users_sorted(self, sample_csv_file):
        """Test that users are returned in sorted order"""
        users = get_available_users(sample_csv_file)
        assert users == sorted(users)

class TestGetUserDetails:
    """Tests for get_user_details function"""

    def test_get_user_details_returns_dict(self):
        """Test that function returns a user details dictionary"""
        user_details = get_user_details('src/data/sd254_users.csv', user_id=0)
        assert isinstance(user_details, dict)

    def test_get_user_details_valid_user(self):
        """Test fetching details for an existing user"""
        user_details = get_user_details('src/data/sd254_users.csv', user_id=1)
        assert user_details['User ID'] == 1
        assert user_details['Person'] == 'Sasha Sadr'

    def test_get_user_details_invalid_user(self):
        """Test that missing user details raises ValueError"""
        with pytest.raises(ValueError, match="No user details found for user"):
            get_user_details('src/data/sd254_users.csv', user_id=999)


# Made with Bob
