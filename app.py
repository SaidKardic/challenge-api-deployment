from typing import Union, Literal, Optional
from preprocessing.cleaning_data import preprocess
from predict.prediction import predict
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
# import tracemalloc
# tracemalloc.start()


app = FastAPI()
json_data = {
  "area": 80,
  "property_type": "APARTMENT",
  "rooms_number": 2,
  "zip_code": 1050,
  "land_area": 100,
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
    zip_code: int
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
async def make_prediction(json_data: Input):
    processed_data = preprocess(json_data)
    prediction = predict(processed_data)
    return prediction



print(make_prediction(json_data))

# async def main():
#     result = await make_prediction(json_data)
#     print(result)

# asyncio.run(main())