from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO, StringIO
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
import sklearn

app = FastAPI()

with open('model_data.pickle', 'rb') as f:
    model_data = pickle.load(f)

weights = model_data['weights']
intercept = model_data['intercept']
encoder= model_data['encoder']

class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    engine: float
    max_power: float
    seats: float
    torque: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
async def predict_item(item: Item) -> float:

    item_dict = item.dict()
    X_new = pd.DataFrame([item_dict])

    X_encoded = encoder.transform(X_new[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    encoded_feature_names = encoder.get_feature_names_out(['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)
    X_new = X_new.drop(['fuel', 'seller_type', 'transmission', 'owner', 'seats'], axis=1)
    X_new = pd.concat([X_new, X_encoded_df], axis=1)

    predicted_price = X_new.values.dot(weights) + intercept

    return predicted_price


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> StreamingResponse:
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    for column in ['fuel', 'seller_type', 'transmission', 'owner', 'seats']:
        if column not in df.columns:
            return {"error": f"Column {column} is missing from the input file."}


    X_encoded = encoder.transform(df[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    encoded_feature_names = encoder.get_feature_names_out(['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)
    X_new = df.drop(['fuel', 'seller_type', 'transmission', 'owner', 'seats'], axis=1)
    X_new = pd.concat([X_new, X_encoded_df], axis=1)

    predictions = X_new.values.dot(weights) + intercept
    df["predictions"] = predictions
    stream = StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv" )
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response