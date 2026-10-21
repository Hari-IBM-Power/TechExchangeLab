"""Request model for the POST /predict endpoint."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr

from config import SAVINGS_PLAN_FIELD_TO_COLUMN


class PredictRequest(BaseModel):
    """Named, strictly typed inputs for the savings plan classifier."""

    model_config = ConfigDict(extra="forbid")

    current_age: StrictInt
    retirement_age: StrictInt
    birth_year: StrictInt
    birth_month: StrictInt
    gender: StrictStr
    city: StrictStr
    state: StrictStr
    zipcode: StrictStr
    per_capita_income_zipcode: StrictInt | StrictFloat
    yearly_income_person: StrictInt | StrictFloat
    total_debt: StrictInt | StrictFloat
    fico_score: StrictInt
    savings_plan: Optional[StrictStr] = None

    def to_model_row(self) -> dict[str, object]:
        """Return the fields keyed by the model's column names."""
        values = self.model_dump()
        return {
            column: values[field]
            for field, column in SAVINGS_PLAN_FIELD_TO_COLUMN.items()
        }
