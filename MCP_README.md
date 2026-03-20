# MCP Server for User Savings Prediction

This MCP (Model Context Protocol) server exposes the user savings prediction functionality as a tool that can be used by AI assistants and other MCP clients.

## Overview

The MCP server wraps the `main()` function from `main.py` and exposes it as an MCP tool called `predict_user_savings`. This allows AI assistants to predict next month's savings for users using Exponential Smoothing time series analysis.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the transaction data file:
   - `transactions_2025_salary_filtered.csv` should be in the project root

## Running the MCP Server

### Standalone Mode
Run the server directly:
```bash
python mcp_server.py
```

### Integration with Claude Desktop or Other MCP Clients

Add the following configuration to your MCP client settings (e.g., Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "user-savings-prediction": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env": {}
    }
  }
}
```

Replace `/absolute/path/to/` with the actual path to your project directory.

## Available Tools

### `predict_user_savings`

Predicts next month's savings for a specified user.

**Parameters:**
- `user_id` (integer, required): The user ID to predict savings for (e.g., 1, 2, 3, 4, 5)

**Behavior:**
- If a trained model exists for the user, it loads and uses it for prediction
- If no model exists but the user has historical data, it automatically trains a new model
- Returns an error if the user doesn't exist in the dataset

**Returns:**
- Predicted next month's savings
- Last month's savings and balance
- Historical statistics (average, standard deviation)
- Model information and metadata

**Example Usage (via MCP client):**
```json
{
  "tool": "predict_user_savings",
  "arguments": {
    "user_id": 1
  }
}
```

## Features

- ✅ Automatic model training if model doesn't exist
- ✅ Model caching for faster predictions
- ✅ Comprehensive error handling
- ✅ Detailed prediction results with historical context
- ✅ Support for multiple users

## Available Users

The system currently supports users with IDs 1-5 (based on the pre-trained models in the `src/` directory). New users can be added if they exist in the transaction data.

## Project Structure

```
.
├── mcp_server.py                          # MCP server implementation
├── main.py                                # Main prediction logic
├── requirements.txt                       # Python dependencies
├── transactions_2025_salary_filtered.csv  # Transaction data
├── src/
│   ├── data_engineering.py               # Data processing functions
│   ├── ml_model.py                       # ML model training/prediction
│   └── model_user_*.pkl                  # Pre-trained models
└── MCP_README.md                         # This file
```

## Troubleshooting

### Import Errors
If you see import errors for `mcp` modules, ensure you've installed the MCP SDK:
```bash
pip install mcp>=1.0.0
```

### User Not Found
If you get a "User not found" error, check:
1. The user ID exists in `transactions_2025_salary_filtered.csv`
2. The user has sufficient historical data for training

### Model Training Issues
If model training fails:
1. Ensure the user has at least 3 months of transaction data
2. Check that the CSV file is properly formatted
3. Verify all dependencies are installed

## Development

To modify the MCP server:
1. Edit `mcp_server.py` to add new tools or modify existing ones
2. The server uses the `@app.list_tools()` decorator to register tools
3. The `@app.call_tool()` decorator handles tool execution

## License

This project is part of the TCC Timeseries prediction system.