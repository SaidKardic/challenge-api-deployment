from typing import Literal, Optional
from preprocessing.cleaning_data import preprocess
from predict.prediction import predict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder

app = FastAPI()

class Input(BaseModel):
    area: int
    property_type: Literal["APARTMENT", "HOUSE"]
    rooms_number: int
    zip_code: int = Field(ge=1000, le=9999)
    land_area: int 
    garden: Optional[bool]
    garden_area: Optional[int]
    equipped_kitchen: Optional[bool]
    swimming_pool: Optional[bool]
    furnished: Optional[bool]
    open_fire: Optional[bool]
    terrace: Optional[bool]
    terrace_area: Optional[int]
    facades_number: Optional[int]
    building_state: Literal["NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD"]

@app.get("/")
async def home():
    return "Alive"

@app.get("/predict")
async def data_format():
    return "House data must be provided in the following format: area: int, property_type: [APARTMENT or HOUSE], rooms_number: int, zip_code: int[between 1000 & 9999], land_area: int, garden: Optional[bool], garden_area: Optional[int], equipped_kitchen: Optional[bool], swimming_pool: Optional[bool], furnished: Optional[bool], open_fire: Optional[bool], terrace: Optional[bool], terrace_area: Optional[int], facades_number: Optional[int], building_state: [NEW, GOOD, TO RENOVATE, JUST RENOVATED, TO REBUILD]"

@app.post("/predict")
async def make_prediction(data: Input):

    json_data = jsonable_encoder(data)
    processed_data = preprocess(json_data)
    prediction = predict(processed_data)
    return prediction
