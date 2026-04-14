"""
Unit tests for train_savings_plan_classifier.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# Try to import sklearn-dependent modules, skip tests if not available
try:
    from src.train_savings_plan_classifier import (
        normalize_text,
        canonical_key,
        standardize_columns,
        clean_dataframe,
        build_pipeline,
        CANONICAL_COLUMNS,
        NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define dummy values for when sklearn is not available
    CANONICAL_COLUMNS = []
    NUMERIC_COLUMNS = []
    CATEGORICAL_COLUMNS = []

# Skip all tests if sklearn is not available
pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")


class TestNormalizeText:
    """Tests for normalize_text function"""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization"""
        assert normalize_text("  Hello World  ") == "Hello World"
    
    def test_normalize_text_multiple_spaces(self):
        """Test normalization of multiple spaces"""
        assert normalize_text("Hello    World") == "Hello World"
    
    def test_normalize_text_en_dash(self):
        """Test replacement of en-dash with hyphen"""
        assert normalize_text("Per Capita Income – Zipcode") == "Per Capita Income - Zipcode"
    
    def test_normalize_text_mixed(self):
        """Test normalization with multiple issues"""
        assert normalize_text("  Hello  –  World  ") == "Hello - World"
    
    def test_normalize_text_empty(self):
        """Test normalization of empty string"""
        assert normalize_text("") == ""


class TestCanonicalKey:
    """Tests for canonical_key function"""
    
    def test_canonical_key_lowercase(self):
        """Test conversion to lowercase"""
        assert canonical_key("Current Age") == "current age"
    
    def test_canonical_key_with_normalization(self):
        """Test canonical key with normalization"""
        assert canonical_key("  Current  Age  ") == "current age"
    
    def test_canonical_key_with_dash(self):
        """Test canonical key with dash replacement"""
        assert canonical_key("Per Capita Income – Zipcode") == "per capita income - zipcode"


class TestStandardizeColumns:
    """Tests for standardize_columns function"""
    
    def test_standardize_columns_exact_match(self):
        """Test standardization with exact column names"""
        df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        result = standardize_columns(df)
        assert list(result.columns) == CANONICAL_COLUMNS
    
    def test_standardize_columns_lowercase(self):
        """Test standardization with lowercase column names"""
        df = pd.DataFrame(columns=[col.lower() for col in CANONICAL_COLUMNS])
        result = standardize_columns(df)
        assert list(result.columns) == CANONICAL_COLUMNS
    
    def test_standardize_columns_with_aliases(self):
        """Test standardization with column aliases"""
        df = pd.DataFrame(columns=[
            "current age", "retirement age", "birth year", "birth month",
            "gender", "city", "state", "zip code",
            "per capita income - zipcode", "yearly income - person",
            "total debt", "fico score", "savings plan"
        ])
        result = standardize_columns(df)
        assert list(result.columns) == CANONICAL_COLUMNS
    
    def test_standardize_columns_missing_required(self):
        """Test error when required columns are missing"""
        df = pd.DataFrame(columns=["Current Age", "Gender"])
        with pytest.raises(ValueError, match="Missing required columns"):
            standardize_columns(df)
    
    def test_standardize_columns_extra_columns(self):
        """Test that extra columns are removed"""
        columns = CANONICAL_COLUMNS + ["Extra Column"]
        df = pd.DataFrame(columns=columns)
        result = standardize_columns(df)
        assert "Extra Column" not in result.columns
        assert list(result.columns) == CANONICAL_COLUMNS


class TestCleanDataframe:
    """Tests for clean_dataframe function"""
    
    def test_clean_dataframe_removes_empty_savings_plan(self):
        """Test removal of rows with empty Savings Plan"""
        df = pd.DataFrame({
            'Current Age': [25, 30, 35],
            'Retirement Age': [65, 65, 65],
            'Birth Year': [1999, 1994, 1989],
            'Birth Month': [1, 2, 3],
            'Gender': ['Male', 'Female', 'Male'],
            'City': ['NYC', 'LA', 'Chicago'],
            'State': ['NY', 'CA', 'IL'],
            'Zipcode': ['10001', '90001', '60601'],
            'Per Capita Income - Zipcode': [75000, 85000, 70000],
            'Yearly Income - Person': [120000, 150000, 100000],
            'Total Debt': [50000, 30000, 40000],
            'FICO Score': [720, 750, 680],
            'Savings Plan': ['Plan A', '', 'Plan B']
        })
        result = clean_dataframe(df)
        assert len(result) == 2
        assert '' not in result['Savings Plan'].values
    
    def test_clean_dataframe_removes_nan_savings_plan(self):
        """Test removal of rows with NaN Savings Plan"""
        df = pd.DataFrame({
            'Current Age': [25, 30],
            'Retirement Age': [65, 65],
            'Birth Year': [1999, 1994],
            'Birth Month': [1, 2],
            'Gender': ['Male', 'Female'],
            'City': ['NYC', 'LA'],
            'State': ['NY', 'CA'],
            'Zipcode': ['10001', '90001'],
            'Per Capita Income - Zipcode': [75000, 85000],
            'Yearly Income - Person': [120000, 150000],
            'Total Debt': [50000, 30000],
            'FICO Score': [720, 750],
            'Savings Plan': ['Plan A', 'nan']
        })
        result = clean_dataframe(df)
        assert len(result) == 1
    
    def test_clean_dataframe_converts_numeric_columns(self):
        """Test conversion of numeric columns"""
        df = pd.DataFrame({
            'Current Age': ['25', '30'],
            'Retirement Age': ['65', '65'],
            'Birth Year': ['1999', '1994'],
            'Birth Month': ['1', '2'],
            'Gender': ['Male', 'Female'],
            'City': ['NYC', 'LA'],
            'State': ['NY', 'CA'],
            'Zipcode': ['10001', '90001'],
            'Per Capita Income - Zipcode': ['75000', '85000'],
            'Yearly Income - Person': ['120000', '150000'],
            'Total Debt': ['50000', '30000'],
            'FICO Score': ['720', '750'],
            'Savings Plan': ['Plan A', 'Plan B']
        })
        result = clean_dataframe(df)
        for col in NUMERIC_COLUMNS:
            assert pd.api.types.is_numeric_dtype(result[col])
    
    def test_clean_dataframe_validates_birth_month(self):
        """Test validation of Birth Month range"""
        df = pd.DataFrame({
            'Current Age': [25, 30, 35],
            'Retirement Age': [65, 65, 65],
            'Birth Year': [1999, 1994, 1989],
            'Birth Month': [1, 13, 0],  # Invalid months
            'Gender': ['Male', 'Female', 'Male'],
            'City': ['NYC', 'LA', 'Chicago'],
            'State': ['NY', 'CA', 'IL'],
            'Zipcode': ['10001', '90001', '60601'],
            'Per Capita Income - Zipcode': [75000, 85000, 70000],
            'Yearly Income - Person': [120000, 150000, 100000],
            'Total Debt': [50000, 30000, 40000],
            'FICO Score': [720, 750, 680],
            'Savings Plan': ['Plan A', 'Plan B', 'Plan C']
        })
        result = clean_dataframe(df)
        assert pd.isna(result.iloc[1]['Birth Month'])
        assert pd.isna(result.iloc[2]['Birth Month'])
        assert result.iloc[0]['Birth Month'] == 1
    
    def test_clean_dataframe_validates_fico_score(self):
        """Test validation of FICO Score range"""
        df = pd.DataFrame({
            'Current Age': [25, 30, 35],
            'Retirement Age': [65, 65, 65],
            'Birth Year': [1999, 1994, 1989],
            'Birth Month': [1, 2, 3],
            'Gender': ['Male', 'Female', 'Male'],
            'City': ['NYC', 'LA', 'Chicago'],
            'State': ['NY', 'CA', 'IL'],
            'Zipcode': ['10001', '90001', '60601'],
            'Per Capita Income - Zipcode': [75000, 85000, 70000],
            'Yearly Income - Person': [120000, 150000, 100000],
            'Total Debt': [50000, 30000, 40000],
            'FICO Score': [720, 200, 900],  # Invalid FICO scores
            'Savings Plan': ['Plan A', 'Plan B', 'Plan C']
        })
        result = clean_dataframe(df)
        assert pd.isna(result.iloc[1]['FICO Score'])
        assert pd.isna(result.iloc[2]['FICO Score'])
        assert result.iloc[0]['FICO Score'] == 720
    
    def test_clean_dataframe_validates_age_ranges(self):
        """Test validation of age ranges"""
        df = pd.DataFrame({
            'Current Age': [25, 10, 150],  # Invalid ages
            'Retirement Age': [65, 65, 65],
            'Birth Year': [1999, 1994, 1989],
            'Birth Month': [1, 2, 3],
            'Gender': ['Male', 'Female', 'Male'],
            'City': ['NYC', 'LA', 'Chicago'],
            'State': ['NY', 'CA', 'IL'],
            'Zipcode': ['10001', '90001', '60601'],
            'Per Capita Income - Zipcode': [75000, 85000, 70000],
            'Yearly Income - Person': [120000, 150000, 100000],
            'Total Debt': [50000, 30000, 40000],
            'FICO Score': [720, 750, 680],
            'Savings Plan': ['Plan A', 'Plan B', 'Plan C']
        })
        result = clean_dataframe(df)
        assert pd.isna(result.iloc[1]['Current Age'])
        assert pd.isna(result.iloc[2]['Current Age'])
        assert result.iloc[0]['Current Age'] == 25


class TestBuildPipeline:
    """Tests for build_pipeline function"""
    
    def test_build_pipeline_returns_pipeline(self):
        """Test that build_pipeline returns a Pipeline object"""
        from sklearn.pipeline import Pipeline
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)
    
    def test_build_pipeline_has_preprocessor(self):
        """Test that pipeline has preprocessor step"""
        pipeline = build_pipeline()
        assert 'preprocessor' in pipeline.named_steps
    
    def test_build_pipeline_has_model(self):
        """Test that pipeline has model step"""
        pipeline = build_pipeline()
        assert 'model' in pipeline.named_steps
    
    def test_build_pipeline_model_type(self):
        """Test that model is LogisticRegression"""
        from sklearn.linear_model import LogisticRegression
        pipeline = build_pipeline()
        assert isinstance(pipeline.named_steps['model'], LogisticRegression)
    
    def test_build_pipeline_preprocessor_transformers(self):
        """Test that preprocessor has numeric and categorical transformers"""
        pipeline = build_pipeline()
        preprocessor = pipeline.named_steps['preprocessor']
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert 'num' in transformer_names
        assert 'cat' in transformer_names


class TestMainFunction:
    """Tests for main function"""
    
    @patch('src.train_savings_plan_classifier.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.train_savings_plan_classifier.pickle.dump')
    def test_main_saves_model(self, mock_pickle_dump, mock_file, mock_read_csv):
        """Test that main function saves the trained model"""
        from src.train_savings_plan_classifier import main
        
        # Create mock dataframe with enough samples for train_test_split
        # Need at least 10 samples (5 per class) for 80/20 split to work
        mock_df = pd.DataFrame({
            'Current Age': [25, 30, 35, 40, 45, 28, 32, 38, 42, 48, 26, 31, 36, 41, 46],
            'Retirement Age': [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65],
            'Birth Year': [1999, 1994, 1989, 1984, 1979, 1996, 1992, 1986, 1982, 1976, 1998, 1993, 1988, 1983, 1978],
            'Birth Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
            'City': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
            'State': ['NY', 'CA', 'IL', 'TX', 'AZ', 'NY', 'CA', 'IL', 'TX', 'AZ', 'NY', 'CA', 'IL', 'TX', 'AZ'],
            'Zipcode': ['10001', '90001', '60601', '77001', '85001', '10002', '90002', '60602', '77002', '85002', '10003', '90003', '60603', '77003', '85003'],
            'Per Capita Income - Zipcode': [75000, 85000, 70000, 65000, 60000, 76000, 86000, 71000, 66000, 61000, 77000, 87000, 72000, 67000, 62000],
            'Yearly Income - Person': [120000, 150000, 100000, 90000, 80000, 125000, 155000, 105000, 95000, 85000, 130000, 160000, 110000, 100000, 90000],
            'Total Debt': [50000, 30000, 40000, 35000, 45000, 52000, 32000, 42000, 37000, 47000, 54000, 34000, 44000, 39000, 49000],
            'FICO Score': [720, 750, 680, 700, 690, 725, 755, 685, 705, 695, 730, 760, 690, 710, 700],
            'Savings Plan': ['Plan A', 'Plan B', 'Plan A', 'Plan B', 'Plan A', 'Plan B', 'Plan A', 'Plan B', 'Plan A', 'Plan B', 'Plan A', 'Plan B', 'Plan A', 'Plan B', 'Plan A']
        })
        mock_read_csv.return_value = mock_df
        
        # Run main function
        main()
        
        # Verify model was saved
        mock_pickle_dump.assert_called_once()
    
    @patch('src.train_savings_plan_classifier.pd.read_csv')
    def test_main_handles_missing_file(self, mock_read_csv):
        """Test that main function handles missing data file"""
        from src.train_savings_plan_classifier import main
        
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            main()


# Made with Bob