from typing import Union, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    area: int
    property_type: Literal["APARTMENT", "HOUSE"]
    rooms_number: int
    zip_code: int
    land_area: Optional[int] | None=None
    garden: Optional[bool] | None=None
    garden_area: Optional[int] | None=None
    equipped_kitchen: Optional[bool] | None=None
    swimming_pool: Optional[bool] | None=None
    furnished: Optional[bool] | None=None
    open_fire: Optional[bool] | None=None
    terrace: Optional[bool] | None=None
    terrace_area: Optional[int] | None=None
    facades_number: Optional[int] | None=None
    building_state: Optional[Literal["NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD"]] | None=None

@app.get("/")
async def home():
    return "Alive"

@app.get("/predict")
async def data_format():
    return "House data must be provided in the following format: area: int, property_type: [APARTMENT or HOUSE], rooms_number: int, zip_code: int, land_area: Optional[int, garden: Optional[bool], garden_area: Optional[int], equipped_kitchen: Optional[bool], swimming_pool: Optional[bool], furnished: Optional[bool], open_fire: Optional[bool], terrace: Optional[bool], terrace_area: Optional[int], facades_number: Optional[int], building_state: Optional[NEW, GOOD, TO RENOVATE, JUST RENOVATED, TO REBUILD]"

@app.post("/predict")
async def make_prediction(json_data: Input):
    processed_data = preprocess(json_data)
    prediction = predict(processed_data)
    return prediction