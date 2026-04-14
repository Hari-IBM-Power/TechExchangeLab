"""
Main Orchestration Script for User Savings Prediction
Takes user ID as input and predicts next month's savings using Exponential Smoothing
Automatically trains model if user exists in data but model doesn't exist
"""

import sys
import os
from src.data_engineering import (
    get_user_savings_timeseries,
    validate_user_exists,
    get_available_users,
    get_user_details
)
from src.ml_model import train_and_save_model, load_and_predict


def predict_user_savings(
    user_id: int,
    data_filepath: str = 'src/data/transactions_2025_salary_filtered.csv',
    user_filepath: str = 'src/data/sd254_users.csv'
) -> dict:
    """
    Predict next month's savings for a user
    - If model exists: load and predict
    - If model doesn't exist but user has data: train model and predict
    - If user doesn't exist in data: return error
    
    Parameters:
    -----------
    user_id : int
        User ID to predict savings for
    data_filepath : str
        Path to the transaction data CSV file
    user_filepath : str
        Path to the user details CSV file
        
    Returns:
    --------
    dict
        Dictionary containing prediction results and metadata
        
    Raises:
    -------
    ValueError
        If user doesn't exist in dataset
    """
    print(f"\n{'='*60}")
    print(f"SAVINGS PREDICTION FOR USER {user_id}")
    print(f"{'='*60}\n")
    
    # Validate user exists in dataset
    if not validate_user_exists(data_filepath, user_id):
        available_users = get_available_users(data_filepath)
        raise ValueError(f"User {user_id} is not trained. User not found in dataset. Available users: {available_users}")
    
    print(f"✓ User {user_id} found in dataset")
    
    # Check if model exists
    model_filepath = f"src/models/model_user_{user_id}.pkl"
    model_exists = os.path.exists(model_filepath)
    
    monthly_df = None
    next_month_savings = None

    if model_exists:
        # Load existing model
        print(f"✓ Model found: {model_filepath}")
        try:
            next_month_savings, forecast = load_and_predict(model_filepath, steps=1)
            print(f"✓ Model loaded successfully")
            
            # Load historical data for context
            ts_data, monthly_df = get_user_savings_timeseries(data_filepath, user_id)
            
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print(f"→ Retraining model...")
            model_exists = False
    
    if not model_exists:
        # Train new model
        print(f"✗ Model not found for user {user_id}")
        print(f"→ Training new model...")
        
        try:
            # Get time series data
            ts_data, monthly_df = get_user_savings_timeseries(data_filepath, user_id)
            
            print(f"✓ Loaded {len(ts_data)} months of data")
            print(
                f"  Date range: {monthly_df.iloc[0]['Date'].strftime('%Y-%m')} "
                f"to {monthly_df.iloc[-1]['Date'].strftime('%Y-%m')}"
            )
            
            # Train and save model
            predictor, saved_path = train_and_save_model(ts_data, user_id, save_dir='src/models')
            
            print(f"✓ Model trained and saved to {saved_path}")
            
            # Make prediction
            next_month_savings, forecast = predictor.predict(steps=1)
            
            # Get model summary
            try:
                summary = predictor.get_model_summary()
                print(f"✓ Model AIC: {summary['aic']:.2f}")
            except:
                pass
                
        except Exception as e:
            raise ValueError(f"Error training model for user {user_id}: {e}")
    
    if monthly_df is None or next_month_savings is None:
        raise ValueError(f"Incomplete prediction data for user {user_id}")

    # Prepare results
    last_month = monthly_df.iloc[-1]
    last_month_date = last_month['Date'].strftime('%Y-%m')
    last_month_savings = last_month['Savings']
    last_month_balance = last_month['EndBalance']
    
    # Calculate statistics
    savings_values = monthly_df['Savings'].tolist()
    avg_savings = sum(savings_values) / len(savings_values)
    std_savings = monthly_df['Savings'].astype(float).std()
    
    # Display results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"\nLast Month ({last_month_date}):")
    print(f"  Savings: ₹{last_month_savings:,.2f}")
    print(f"  End Balance: ₹{last_month_balance:,.2f}")
    print(f"\nHistorical Statistics:")
    print(f"  Average Monthly Savings: ₹{avg_savings:,.2f}")
    print(f"  Std Dev: ₹{std_savings:,.2f}")
    print(f"\n{'─'*60}")
    print(f"PREDICTED NEXT MONTH SAVINGS: ₹{next_month_savings:,.2f}")
    print(f"{'─'*60}\n")
    
    user_details = get_user_details(user_filepath, user_id)

    # Prepare return dictionary
    results = {
        'user_id': user_id,
        'user_details': user_details,
        'predicted_next_month_savings': float(next_month_savings),
        'last_month_savings': float(last_month_savings),
        'last_month_balance': float(last_month_balance),
        'last_month_date': last_month_date,
        'average_savings': avg_savings,
        'std_savings': std_savings,
        'n_months': len(monthly_df),
        'model_filepath': model_filepath
    }
    
    return results


def main(user_id: int) -> dict:
    """
    Main function to run the prediction pipeline
    
    Parameters:
    -----------
    user_id : int
        User ID to predict savings for
        
    Returns:
    --------
    dict
        Prediction results or error information
    """
    print("\n" + "="*60)
    print("USER SAVINGS PREDICTION SYSTEM")
    print("Using Exponential Smoothing Time Series Analysis")
    print("="*60)
    
    try:
        # Run prediction
        results = predict_user_savings(user_id)
        
        print(f"✓ Prediction completed successfully!")
        
        return results
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return {
            'error': str(e),
            'user_id': user_id,
            'success': False
        }
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'user_id': user_id,
            'success': False
        }


if __name__ == "__main__":
    # Example usage - replace with desired user_id
    result = main(6)
    print(f"\nReturned result: {result}")

# Made with Bob
