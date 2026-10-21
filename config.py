"""Central configuration for the savings prediction API."""

# Path to the trained savings plan classifier
SAVINGS_PLAN_MODEL_PATH = "./src/models/savings_plan_classifier.pkl"

# Request field name -> model column name, in the order the model was trained on.
# "savings_plan" is optional in the request and ignored by the model pipeline.
SAVINGS_PLAN_FIELD_TO_COLUMN = {
    "current_age": "Current Age",
    "retirement_age": "Retirement Age",
    "birth_year": "Birth Year",
    "birth_month": "Birth Month",
    "gender": "Gender",
    "city": "City",
    "state": "State",
    "zipcode": "Zipcode",
    "per_capita_income_zipcode": "Per Capita Income - Zipcode",
    "yearly_income_person": "Yearly Income - Person",
    "total_debt": "Total Debt",
    "fico_score": "FICO Score",
    "savings_plan": "Savings Plan",
}
