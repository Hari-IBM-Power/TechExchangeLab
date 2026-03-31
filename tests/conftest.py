"""
Pytest configuration and shared fixtures
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing"""
    data = {
        'User': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        'Card': ['1234'] * 12,
        'Year': [2025] * 12,
        'Month': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        'Day': [1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15],
        'Time': ['10:00'] * 12,
        'Amount': [5000, 2000, 5000, 1500, 5000, 2500, 6000, 1000, 6000, 1200, 6000, 1500],
        'Cr_Dr': ['Cr', 'Dr', 'Cr', 'Dr', 'Cr', 'Dr', 'Cr', 'Dr', 'Cr', 'Dr', 'Cr', 'Dr'],
        'Balance': [5000, 3000, 8000, 6500, 11500, 9000, 6000, 5000, 11000, 9800, 15800, 14300]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_transaction_data, tmp_path):
    """Create a temporary CSV file with sample data"""
    csv_path = tmp_path / "test_transactions.csv"
    sample_transaction_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_timeseries_data():
    """Create sample time series data for ML model testing"""
    dates = pd.date_range(start='2025-01-01', periods=12, freq='MS')
    savings = [5000, 6000, 5500, 7000, 6500, 7500, 8000, 7200, 8500, 9000, 8800, 9200]
    ts_data = pd.Series(savings, index=dates)
    return ts_data


@pytest.fixture
def sample_monthly_df():
    """Create sample monthly dataframe"""
    data = {
        'Year': [2025] * 12,
        'Month': list(range(1, 13)),
        'Savings': [5000, 6000, 5500, 7000, 6500, 7500, 8000, 7200, 8500, 9000, 8800, 9200],
        'EndBalance': [5000, 11000, 16500, 23500, 30000, 37500, 45500, 52700, 61200, 70200, 79000, 88200],
        'StartBalance': [0, 5000, 11000, 16500, 23500, 30000, 37500, 45500, 52700, 61200, 70200, 79000],
        'Date': pd.date_range(start='2025-01-01', periods=12, freq='MS')
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture
def mock_flask_app():
    """Create a test Flask app instance"""
    from app import app
    app.config['TESTING'] = True
    return app


@pytest.fixture
def flask_client(mock_flask_app):
    """Create a test client for Flask app"""
    return mock_flask_app.test_client()

# Made with Bob
