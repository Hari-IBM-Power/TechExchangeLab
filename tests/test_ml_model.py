"""
Unit tests for ml_model module
"""
import os

import numpy as np
import pandas as pd
import pytest
from src.ml_model import (
    SavingsPredictor,
    train_and_save_model,
    load_and_predict
)


class TestSavingsPredictorInit:
    """Tests for SavingsPredictor initialization"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        predictor = SavingsPredictor()
        assert predictor.seasonal_periods == 12
        assert predictor.trend == 'add'
        assert predictor.seasonal == 'add'
        assert predictor.use_boxcox is False
        assert predictor.model is None
        assert predictor.fitted_model is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        predictor = SavingsPredictor(
            seasonal_periods=6,
            trend='mul',
            seasonal='mul',
            use_boxcox=True
        )
        assert predictor.seasonal_periods == 6
        assert predictor.trend == 'mul'
        assert predictor.seasonal == 'mul'
        assert predictor.use_boxcox is True


class TestSavingsPredictorTrain:
    """Tests for SavingsPredictor training"""

    def test_train_with_sufficient_data(self, sample_timeseries_data):
        """Test training with sufficient data (12 months)"""
        predictor = SavingsPredictor()
        result = predictor.train(sample_timeseries_data)

        assert result is predictor  # Should return self
        assert predictor.fitted_model is not None
        assert predictor.training_data is not None
        assert len(predictor.training_data) == len(sample_timeseries_data)

    def test_train_with_insufficient_data(self):
        """Test training with insufficient data (< 24 months)"""
        # Create small dataset (6 months)
        dates = pd.date_range(start='2025-01-01', periods=6, freq='MS')
        savings = [5000, 6000, 5500, 7000, 6500, 7500]
        ts_data = pd.Series(savings, index=dates)

        predictor = SavingsPredictor()
        predictor.train(ts_data)

        # Should still train successfully with simpler model
        assert predictor.fitted_model is not None

    def test_train_preserves_original_data(self, sample_timeseries_data):
        """Test that training preserves the original data"""
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)

        pd.testing.assert_series_equal(
            predictor.training_data,
            sample_timeseries_data
        )

    def test_train_with_very_small_data(self):
        """Test training with very small dataset (3 months)"""
        dates = pd.date_range(start='2025-01-01', periods=3, freq='MS')
        savings = [5000, 6000, 5500]
        ts_data = pd.Series(savings, index=dates)

        predictor = SavingsPredictor()
        predictor.train(ts_data)

        # Should train with simple model
        assert predictor.fitted_model is not None


class TestSavingsPredictorPredict:
    """Tests for SavingsPredictor prediction"""

    def test_predict_single_step(self, sample_timeseries_data):
        """Test prediction for single step ahead"""
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)

        next_month, forecast = predictor.predict(steps=1)

        assert isinstance(next_month, (int, float, np.number))
        assert isinstance(forecast, (pd.Series, np.ndarray))
        assert next_month > 0  # Savings should be positive

    def test_predict_multiple_steps(self, sample_timeseries_data):
        """Test prediction for multiple steps ahead"""
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)

        next_month, forecast = predictor.predict(steps=3)

        assert len(forecast) == 3
        assert isinstance(next_month, (int, float, np.number))

    def test_predict_without_training(self):
        """Test that prediction without training raises error"""
        predictor = SavingsPredictor()

        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict(steps=1)

    def test_predict_returns_reasonable_values(self, sample_timeseries_data):
        """Test that predictions are in reasonable range"""
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)

        next_month, _ = predictor.predict(steps=1)

        # Prediction should be within reasonable range of historical data
        mean_savings = sample_timeseries_data.mean()
        std_savings = sample_timeseries_data.std()

        # Prediction should be within 3 standard deviations
        assert mean_savings - 3*std_savings < next_month < mean_savings + 3*std_savings


class TestSavingsPredictorModelSummary:
    """Tests for SavingsPredictor model summary"""

    def test_get_model_summary(self, sample_timeseries_data):
        """Test getting model summary"""
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)

        summary = predictor.get_model_summary()

        assert isinstance(summary, dict)
        assert 'aic' in summary
        assert 'bic' in summary
        assert 'aicc' in summary
        assert 'params' in summary
        assert 'n_observations' in summary

    def test_get_summary_without_training(self):
        """Test that getting summary without training raises error"""
        predictor = SavingsPredictor()

        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.get_model_summary()

    def test_summary_n_observations_correct(self, sample_timeseries_data):
        """Test that summary reports correct number of observations"""
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)

        summary = predictor.get_model_summary()
        assert summary['n_observations'] == len(sample_timeseries_data)


class TestSavingsPredictorSaveLoad:
    """Tests for SavingsPredictor save and load"""

    def test_save_model(self, sample_timeseries_data, temp_model_dir):
        """Test saving a trained model"""
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)

        filepath = os.path.join(temp_model_dir, 'test_model.pkl')
        predictor.save_model(filepath)

        assert os.path.exists(filepath)

    def test_save_without_training(self, temp_model_dir):
        """Test that saving without training raises error"""
        predictor = SavingsPredictor()
        filepath = os.path.join(temp_model_dir, 'test_model.pkl')

        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.save_model(filepath)

    def test_load_model(self, sample_timeseries_data, temp_model_dir):
        """Test loading a saved model"""
        # Train and save
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)
        filepath = os.path.join(temp_model_dir, 'test_model.pkl')
        predictor.save_model(filepath)

        # Load
        loaded_predictor = SavingsPredictor.load_model(filepath)

        assert loaded_predictor.fitted_model is not None
        assert loaded_predictor.training_data is not None
        assert loaded_predictor.seasonal_periods == predictor.seasonal_periods

    def test_loaded_model_can_predict(self, sample_timeseries_data, temp_model_dir):
        """Test that loaded model can make predictions"""
        # Train and save
        predictor = SavingsPredictor()
        predictor.train(sample_timeseries_data)
        filepath = os.path.join(temp_model_dir, 'test_model.pkl')
        predictor.save_model(filepath)

        # Load and predict
        loaded_predictor = SavingsPredictor.load_model(filepath)
        next_month, _forecast = loaded_predictor.predict(steps=1)

        assert isinstance(next_month, (int, float, np.number))
        assert next_month > 0

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            SavingsPredictor.load_model('nonexistent_model.pkl')


class TestTrainAndSaveModel:
    """Tests for train_and_save_model helper function"""

    def test_train_and_save_creates_file(self, sample_timeseries_data, temp_model_dir):
        """Test that function creates model file"""
        _predictor, filepath = train_and_save_model(
            sample_timeseries_data,
            user_id=0,
            save_dir=temp_model_dir
        )

        assert os.path.exists(filepath)
        assert 'model_user_0.pkl' in filepath

    def test_train_and_save_returns_predictor(self, sample_timeseries_data, temp_model_dir):
        """Test that function returns trained predictor"""
        predictor, _filepath = train_and_save_model(
            sample_timeseries_data,
            user_id=0,
            save_dir=temp_model_dir
        )

        assert isinstance(predictor, SavingsPredictor)
        assert predictor.fitted_model is not None

    def test_train_and_save_predictor_can_predict(self, sample_timeseries_data, temp_model_dir):
        """Test that returned predictor can make predictions"""
        predictor, _ = train_and_save_model(
            sample_timeseries_data,
            user_id=0,
            save_dir=temp_model_dir
        )

        next_month, _forecast = predictor.predict(steps=1)
        assert isinstance(next_month, (int, float, np.number))


class TestLoadAndPredict:
    """Tests for load_and_predict helper function"""

    def test_load_and_predict(self, sample_timeseries_data, temp_model_dir):
        """Test loading model and making prediction"""
        # First create a model
        _, filepath = train_and_save_model(
            sample_timeseries_data,
            user_id=0,
            save_dir=temp_model_dir
        )

        # Load and predict
        next_month, forecast = load_and_predict(filepath, steps=1)

        assert isinstance(next_month, (int, float, np.number))
        assert isinstance(forecast, (pd.Series, np.ndarray))

    def test_load_and_predict_multiple_steps(self, sample_timeseries_data, temp_model_dir):
        """Test loading and predicting multiple steps"""
        # Create model
        _, filepath = train_and_save_model(
            sample_timeseries_data,
            user_id=0,
            save_dir=temp_model_dir
        )

        # Load and predict 3 steps
        _next_month, forecast = load_and_predict(filepath, steps=3)

        assert len(forecast) == 3

    def test_load_and_predict_nonexistent_file(self):
        """Test that loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_and_predict('nonexistent_model.pkl', steps=1)


class TestSavingsPredictorFullModel:  # pylint: disable=too-few-public-methods
    """Tests for full seasonal model (line 73 coverage)"""

    def test_train_with_full_seasonal_model(self):
        """Test training with enough data for full seasonal model (24+ months)"""
        # Create dataset with 24 months (2 full years)
        dates = pd.date_range(start='2023-01-01', periods=24, freq='MS')
        savings = [5000 + i*100 for i in range(24)]  # Increasing trend
        ts_data = pd.Series(savings, index=dates)

        predictor = SavingsPredictor()
        predictor.train(ts_data)

        # Should use full model with seasonal component
        assert predictor.fitted_model is not None
        assert len(predictor.training_data) == 24

        # Should be able to predict
        next_month, _forecast = predictor.predict(steps=1)
        assert isinstance(next_month, (int, float, np.number))

# Made with Bob
