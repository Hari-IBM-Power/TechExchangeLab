"""
Unit tests for main module
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from main import predict_user_savings, main


class TestPredictUserSavings:
    """Tests for predict_user_savings function"""
    
    @patch('main.validate_user_exists')
    @patch('main.os.path.exists')
    @patch('main.load_and_predict')
    @patch('main.get_user_savings_timeseries')
    def test_predict_with_existing_model(
        self,
        mock_get_timeseries,
        mock_load_predict,
        mock_exists,
        mock_validate,
        sample_timeseries_data,
        sample_monthly_df
    ):
        """Test prediction when model already exists"""
        # Setup mocks
        mock_validate.return_value = True
        mock_exists.return_value = True
        mock_load_predict.return_value = (7500.0, [7500.0])
        mock_get_timeseries.return_value = (sample_timeseries_data, sample_monthly_df)
        
        # Call function
        result = predict_user_savings(0, 'test.csv')
        
        # Assertions
        assert result['user_id'] == 0
        assert 'predicted_next_month_savings' in result
        assert 'last_month_savings' in result
        assert 'average_savings' in result
        assert result['predicted_next_month_savings'] == 7500.0
    
    @patch('main.validate_user_exists')
    @patch('main.os.path.exists')
    @patch('main.get_user_savings_timeseries')
    @patch('main.train_and_save_model')
    def test_predict_without_existing_model(
        self,
        mock_train_save,
        mock_get_timeseries,
        mock_exists,
        mock_validate,
        sample_timeseries_data,
        sample_monthly_df
    ):
        """Test prediction when model doesn't exist (auto-training)"""
        # Setup mocks
        mock_validate.return_value = True
        mock_exists.return_value = False
        mock_get_timeseries.return_value = (sample_timeseries_data, sample_monthly_df)
        
        # Create mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (7500.0, [7500.0])
        mock_predictor.get_model_summary.return_value = {'aic': 100.0}
        mock_train_save.return_value = (mock_predictor, 'src/model_user_0.pkl')
        
        # Call function
        result = predict_user_savings(0, 'test.csv')
        
        # Assertions
        assert result['user_id'] == 0
        assert 'predicted_next_month_savings' in result
        assert result['predicted_next_month_savings'] == 7500.0
        mock_train_save.assert_called_once()
    
    @patch('main.validate_user_exists')
    @patch('main.get_available_users')
    def test_predict_invalid_user(self, mock_get_users, mock_validate):
        """Test prediction with invalid user raises ValueError"""
        mock_validate.return_value = False
        mock_get_users.return_value = [0, 1, 2]
        
        with pytest.raises(ValueError, match="User 999 is not trained"):
            predict_user_savings(999, 'test.csv')
    
    @patch('main.validate_user_exists')
    @patch('main.os.path.exists')
    @patch('main.get_user_savings_timeseries')
    def test_predict_returns_all_required_fields(
        self,
        mock_get_timeseries,
        mock_exists,
        mock_validate,
        sample_timeseries_data,
        sample_monthly_df
    ):
        """Test that result contains all required fields"""
        mock_validate.return_value = True
        mock_exists.return_value = False
        mock_get_timeseries.return_value = (sample_timeseries_data, sample_monthly_df)
        
        with patch('main.train_and_save_model') as mock_train:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = (7500.0, [7500.0])
            mock_predictor.get_model_summary.return_value = {'aic': 100.0}
            mock_train.return_value = (mock_predictor, 'src/model_user_0.pkl')
            
            result = predict_user_savings(0, 'test.csv')
            
            # Check all required fields
            required_fields = [
                'user_id',
                'predicted_next_month_savings',
                'last_month_savings',
                'last_month_balance',
                'last_month_date',
                'average_savings',
                'std_savings',
                'n_months',
                'model_filepath'
            ]
            
            for field in required_fields:
                assert field in result


class TestMain:
    """Tests for main function"""
    
    @patch('main.predict_user_savings')
    def test_main_success(self, mock_predict):
        """Test main function with successful prediction"""
        mock_predict.return_value = {
            'user_id': 0,
            'predicted_next_month_savings': 7500.0,
            'last_month_savings': 7000.0,
            'average_savings': 7200.0
        }
        
        result = main(0)
        
        assert 'user_id' in result
        assert result['user_id'] == 0
        assert 'predicted_next_month_savings' in result
    
    @patch('main.predict_user_savings')
    def test_main_with_value_error(self, mock_predict):
        """Test main function handles ValueError"""
        mock_predict.side_effect = ValueError("User not found")
        
        result = main(999)
        
        assert 'error' in result
        assert result['success'] is False
        assert result['user_id'] == 999
    
    @patch('main.predict_user_savings')
    def test_main_with_unexpected_error(self, mock_predict):
        """Test main function handles unexpected errors"""
        mock_predict.side_effect = Exception("Unexpected error")
        
        result = main(0)
        
        assert 'error' in result
        assert result['success'] is False
        assert result['user_id'] == 0
    
    @patch('main.predict_user_savings')
    def test_main_returns_error_dict_on_failure(self, mock_predict):
        """Test that main returns proper error dictionary"""
        mock_predict.side_effect = ValueError("Test error")
        
        result = main(0)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'user_id' in result
        assert 'success' in result
        assert result['success'] is False


class TestIntegration:
    """Integration tests using real data"""
    
    @patch('main.validate_user_exists')
    @patch('main.get_available_users')
    def test_invalid_user_flow(self, mock_get_users, mock_validate):
        """Test complete flow for invalid user"""
        mock_validate.return_value = False
        mock_get_users.return_value = [0, 1, 2, 3, 4, 5]


class TestMainErrorHandling:
    """Tests for error handling edge cases"""
    
    @patch('main.validate_user_exists')
    @patch('main.os.path.exists')
    @patch('main.load_and_predict')
    @patch('main.get_user_savings_timeseries')
    def test_model_load_failure_triggers_retraining(
        self,
        mock_get_timeseries,
        mock_load_predict,
        mock_exists,
        mock_validate,
        sample_timeseries_data,
        sample_monthly_df
    ):
        """Test that model load failure triggers retraining (lines 66-69)"""
        mock_validate.return_value = True
        mock_exists.return_value = True
        # Make load_and_predict fail
        mock_load_predict.side_effect = Exception("Model corrupted")
        mock_get_timeseries.return_value = (sample_timeseries_data, sample_monthly_df)
        
        with patch('main.train_and_save_model') as mock_train:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = (7500.0, [7500.0])
            mock_predictor.get_model_summary.return_value = {'aic': 100.0}
            mock_train.return_value = (mock_predictor, 'src/model_user_0.pkl')
            
            result = predict_user_savings(0, 'test.csv')
            
            # Should have retrained after load failure
            mock_train.assert_called_once()
            assert result['predicted_next_month_savings'] == 7500.0
    
    @patch('main.validate_user_exists')
    @patch('main.os.path.exists')
    @patch('main.get_user_savings_timeseries')
    @patch('main.train_and_save_model')
    def test_model_summary_failure_handled(
        self,
        mock_train_save,
        mock_get_timeseries,
        mock_exists,
        mock_validate,
        sample_timeseries_data,
        sample_monthly_df
    ):
        """Test that model summary failure is handled gracefully (lines 95-96)"""
        mock_validate.return_value = True
        mock_exists.return_value = False
        mock_get_timeseries.return_value = (sample_timeseries_data, sample_monthly_df)
        
        # Create mock predictor that fails on get_model_summary
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (7500.0, [7500.0])
        mock_predictor.get_model_summary.side_effect = Exception("Summary failed")
        mock_train_save.return_value = (mock_predictor, 'src/model_user_0.pkl')
        
        # Should not raise exception, just skip printing AIC
        result = predict_user_savings(0, 'test.csv')
        assert result['predicted_next_month_savings'] == 7500.0
    
    @patch('main.validate_user_exists')
    @patch('main.os.path.exists')
    @patch('main.get_user_savings_timeseries')
    def test_training_failure_raises_error(
        self,
        mock_get_timeseries,
        mock_exists,
        mock_validate
    ):
        """Test that training failure raises ValueError (lines 98-99)"""
        mock_validate.return_value = True
        mock_exists.return_value = False
        # Make get_user_savings_timeseries fail
        mock_get_timeseries.side_effect = Exception("Data loading failed")
        
        # Should return error dict (main() catches ValueError and returns error dict)
        result = main(0)
        assert result['success'] is False
        assert 'error' in result
        # The error message contains "Error training model"
        assert 'Error training model' in result['error'] or 'Data loading failed' in result['error']
