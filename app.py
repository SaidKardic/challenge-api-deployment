from typing import Union, Literal, Optional
from preprocessing.cleaning_data import preprocess
from predict.prediction import predict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder

app = FastAPI()
example_data = {
  "area": 80,
  "property_type": "APARTMENT",
  "rooms_number": 2,
  "zip_code": 1050,
  "land_area": 80,
  "garden": False,
  "garden_area": 0,
  "equipped_kitchen": True,
  "swimming_pool": False,
  "furnished": False,
  "open_fire": False,
  "terrace": False,
  "terrace_area": 0,
  "facades_number": 2,
  "building_state": "GOOD"
}
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
    return "House data must be provided in the following format: area: int, property_type: [APARTMENT or HOUSE], rooms_number: int, zip_code: int, land_area: Optional[int, garden: Optional[bool], garden_area: Optional[int], equipped_kitchen: Optional[bool], swimming_pool: Optional[bool], furnished: Optional[bool], open_fire: Optional[bool], terrace: Optional[bool], terrace_area: Optional[int], facades_number: Optional[int], building_state: Optional[NEW, GOOD, TO RENOVATE, JUST RENOVATED, TO REBUILD]"

@app.post("/predict")
async def make_prediction(data: Input):

    json_data = jsonable_encoder(data)
    print("before prepro ", json_data)
    print("before prepro ", type(json_data))
    processed_data = preprocess(json_data)
    print("after prepro ", processed_data)
    for row in processed_data.iterrows():
        print(row)
    print("after prepro ", type(processed_data))
    prediction = predict(processed_data)
    return prediction

print(predict(preprocess(example_data)))
