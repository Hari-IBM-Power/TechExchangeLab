# Test Suite for Time Series Savings Prediction System

This directory contains comprehensive unit tests for all modules in the project.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest fixtures and configuration
├── test_data_engineering.py    # Tests for data_engineering module
├── test_ml_model.py            # Tests for ml_model module
├── test_main.py                # Tests for main orchestration
├── test_app.py                 # Tests for Flask API
└── README.md                   # This file
```

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_data_engineering.py
```

### Run specific test class
```bash
pytest tests/test_data_engineering.py::TestLoadTransactionData
```

### Run specific test function
```bash
pytest tests/test_data_engineering.py::TestLoadTransactionData::test_load_valid_csv
```

### Run tests with coverage
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Skip slow tests
```bash
pytest -m "not slow"
```

## Test Coverage

The test suite covers:

### 1. Data Engineering (`test_data_engineering.py`)
- ✅ CSV file loading
- ✅ Data preprocessing
- ✅ Monthly savings calculation
- ✅ Time series data preparation
- ✅ User validation
- ✅ Available users listing

### 2. ML Model (`test_ml_model.py`)
- ✅ Model initialization
- ✅ Training with various data sizes
- ✅ Prediction (single and multiple steps)
- ✅ Model summary statistics
- ✅ Model save/load functionality
- ✅ Helper functions

### 3. Main Orchestration (`test_main.py`)
- ✅ Prediction with existing models
- ✅ Auto-training when model doesn't exist
- ✅ Invalid user handling
- ✅ Error handling
- ✅ Return value validation

### 4. Flask API (`test_app.py`)
- ✅ POST /savings endpoint
- ✅ GET /health endpoint
- ✅ GET / (documentation) endpoint
- ✅ Input validation
- ✅ Error responses
- ✅ Response format validation

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_transaction_data`: Sample transaction DataFrame
- `sample_csv_file`: Temporary CSV file with sample data
- `sample_timeseries_data`: Sample time series for ML testing
- `sample_monthly_df`: Sample monthly aggregated data
- `temp_model_dir`: Temporary directory for model files
- `flask_client`: Flask test client
- `mock_flask_app`: Mock Flask application

## Writing New Tests

### Test Class Structure
```python
class TestFeatureName:
    """Tests for feature_name functionality"""
    
    def test_specific_behavior(self, fixture_name):
        """Test that specific behavior works correctly"""
        # Arrange
        input_data = ...
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_value
```

### Using Fixtures
```python
def test_with_fixture(sample_csv_file):
    """Test using a fixture"""
    df = load_transaction_data(sample_csv_file)
    assert not df.empty
```

### Mocking External Dependencies
```python
from unittest.mock import patch

def test_with_mock():
    """Test with mocked dependency"""
    with patch('module.function') as mock_func:
        mock_func.return_value = expected_value
        result = function_under_test()
        assert result == expected_value
```

### Async Tests
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await async_function()
    assert result is not None
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **One Assert Per Test**: Focus each test on a single behavior
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use Fixtures**: Reuse common setup code
5. **Mock External Dependencies**: Isolate unit tests
6. **Test Edge Cases**: Include boundary conditions and error cases
7. **Keep Tests Fast**: Use mocks to avoid slow operations
8. **Test Independence**: Each test should run independently

## Continuous Integration

To run tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest --cov=src --cov-report=xml
```

## Troubleshooting

### Import Errors
If you see import errors, ensure you're running pytest from the project root:
```bash
cd /path/to/TCC\ Timeseries
pytest
```

### Fixture Not Found
Make sure `conftest.py` is in the tests directory and properly configured.

### Async Test Failures
Ensure `pytest-asyncio` is installed and `asyncio_mode = auto` is set in `pytest.ini`.

## Coverage Goals

Target coverage: **>80%** for all modules

Current coverage can be checked with:
```bash
pytest --cov=src --cov-report=term-missing
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this README if needed