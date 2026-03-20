#!/usr/bin/env python3
"""
MCP Server for User Savings Prediction
Exposes the main() function from main.py as an MCP tool
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import the main function from main.py
from main import main


# Create MCP server instance
app = Server("user-savings-prediction")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available MCP tools
    """
    return [
        Tool(
            name="predict_user_savings",
            description=(
                "Predict next month's savings for a user using Exponential Smoothing time series analysis. "
                "If a trained model exists for the user, it will be loaded and used for prediction. "
                "If no model exists but the user has historical data, a new model will be trained automatically. "
                "Returns predicted savings, historical statistics, and model information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The user ID to predict savings for (e.g., 1, 2, 3, 4, 5)",
                        "minimum": 1
                    }
                },
                "required": ["user_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """
    Handle tool calls
    """
    if name != "predict_user_savings":
        raise ValueError(f"Unknown tool: {name}")
    
    # Extract user_id from arguments
    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a dictionary")
    
    user_id = arguments.get("user_id")
    if user_id is None:
        raise ValueError("user_id is required")
    
    if not isinstance(user_id, int):
        raise ValueError("user_id must be an integer")
    
    # Call the main function
    try:
        result = main(user_id)
        
        # Format the result as a readable text response
        if result.get('success') is False or 'error' in result:
            response_text = f"❌ Error predicting savings for user {user_id}:\n{result.get('error', 'Unknown error')}"
        else:
            response_text = f"""✅ Savings Prediction for User {user_id}

📊 Prediction Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Next Month Savings: ₹{result['predicted_next_month_savings']:,.2f}

📈 Historical Context:
  • Last Month ({result['last_month_date']}):
    - Savings: ₹{result['last_month_savings']:,.2f}
    - End Balance: ₹{result['last_month_balance']:,.2f}
  
  • Historical Statistics ({result['n_months']} months):
    - Average Monthly Savings: ₹{result['average_savings']:,.2f}
    - Standard Deviation: ₹{result['std_savings']:,.2f}

🤖 Model Information:
  • Model Path: {result['model_filepath']}
  • User ID: {result['user_id']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Raw JSON Result:
{json.dumps(result, indent=2)}
"""
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        error_text = f"❌ Unexpected error while predicting savings for user {user_id}:\n{str(e)}"
        return [TextContent(type="text", text=error_text)]


async def main_server():
    """
    Main entry point for the MCP server
    """
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    # Run the MCP server
    asyncio.run(main_server())

# Made with Bob
