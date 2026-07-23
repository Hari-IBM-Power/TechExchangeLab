"""
Machine Learning Module for Time Series Forecasting
Implements Exponential Smoothing for savings prediction
"""

import pickle
import warnings
from typing import Dict, Tuple

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')


class SavingsPredictor:
    """
    Exponential Smoothing model for predicting user savings
    """

    def __init__(self, seasonal_periods: int = 12, trend: str = 'add',
                 seasonal: str = 'add', use_boxcox: bool = False):
        """
        Initialize the Savings Predictor

        Parameters:
        -----------
        seasonal_periods : int
            Number of periods in a season (default: 12 for monthly data)
        trend : str
            Type of trend component ('add', 'mul', or None)
        seasonal : str
            Type of seasonal component ('add', 'mul', or None)
        use_boxcox : bool
            Whether to use Box-Cox transformation
        """
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.use_boxcox = use_boxcox
        self.model = None
        self.fitted_model = None
        self.training_data: pd.Series | None = None

    def train(self, timeseries_data: pd.Series) -> 'SavingsPredictor':
        """
        Train the Exponential Smoothing model

        Parameters:
        -----------
        timeseries_data : pd.Series
            Time series data with datetime index

        Returns:
        --------
        SavingsPredictor
            Trained model instance
        """
        self.training_data = timeseries_data.copy()

        # Determine if we have enough data for seasonal component
        n_obs = len(timeseries_data)

        if n_obs < self.seasonal_periods * 2:
            # Not enough data for seasonal component
            print(f"Warning: Only {n_obs} observations. Using simple exponential smoothing.")
            self.model = ExponentialSmoothing(
                timeseries_data,
                trend=self.trend if n_obs >= 10 else None,
                seasonal=None
            )
        else:
            # Enough data for full model
            self.model = ExponentialSmoothing(
                timeseries_data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )

        # Fit the model
        self.fitted_model = self.model.fit(optimized=True)

        return self

    def predict(self, steps: int = 1) -> Tuple[float, pd.Series]:
        """
        Predict future savings

        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast

        Returns:
        --------
        Tuple[float, pd.Series]
            Next month prediction and full forecast series
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before prediction")

        # Generate forecast
        forecast = self.fitted_model.forecast(steps=steps)

        # Return next month prediction and full forecast
        next_month_prediction = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]

        return next_month_prediction, forecast

    def get_model_summary(self) -> Dict:
        """
        Get model summary statistics

        Returns:
        --------
        Dict
            Dictionary containing model parameters and fit statistics
        """
        if self.fitted_model is None or self.training_data is None:
            raise ValueError("Model must be trained first")

        summary = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'aicc': self.fitted_model.aicc,
            'params': self.fitted_model.params,
            'n_observations': len(self.training_data)
        }

        return summary

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a pickle file

        Parameters:
        -----------
        filepath : str
            Path where the model should be saved
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'fitted_model': self.fitted_model,
            'training_data': self.training_data,
            'seasonal_periods': self.seasonal_periods,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'use_boxcox': self.use_boxcox
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'SavingsPredictor':
        """
        Load a trained model from a pickle file

        Parameters:
        -----------
        filepath : str
            Path to the saved model file

        Returns:
        --------
        SavingsPredictor
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance
        predictor = cls(
            seasonal_periods=model_data['seasonal_periods'],
            trend=model_data['trend'],
            seasonal=model_data['seasonal'],
            use_boxcox=model_data['use_boxcox']
        )

        # Restore model state
        predictor.fitted_model = model_data['fitted_model']
        predictor.training_data = model_data['training_data']

        return predictor


def train_and_save_model(timeseries_data: pd.Series, user_id: int,
                         save_dir: str = 'src') -> Tuple[SavingsPredictor, str]:
    """
    Train a model and save it for a specific user

    Parameters:
    -----------
    timeseries_data : pd.Series
        Time series data for training
    user_id : int
        User ID for naming the model file
    save_dir : str
        Directory to save the model

    Returns:
    --------
    Tuple[SavingsPredictor, str]
        Trained model and filepath where it was saved
    """
    # Initialize and train model
    predictor = SavingsPredictor(seasonal_periods=12, trend='add', seasonal='add')
    predictor.train(timeseries_data)

    # Save model
    filepath = f"{save_dir}/model_user_{user_id}.pkl"
    predictor.save_model(filepath)

    return predictor, filepath


def load_and_predict(model_filepath: str, steps: int = 1) -> Tuple[float, pd.Series]:
    """
    Load a saved model and make predictions

    Parameters:
    -----------
    model_filepath : str
        Path to the saved model file
    steps : int
        Number of steps ahead to forecast

    Returns:
    --------
    Tuple[float, pd.Series]
        Next month prediction and full forecast series
    """
    predictor = SavingsPredictor.load_model(model_filepath)
    next_month, forecast = predictor.predict(steps=steps)

    return next_month, forecast

# Made with Bob
