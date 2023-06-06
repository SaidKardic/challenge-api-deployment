import pickle
import pandas as pd

with open('model/finalized_model.sav', 'rb') as file:
    model = pickle.load(file)

def predict(df):
    y_pred = model.predict(df)
    result = {'Prediction': y_pred[0]}

    return result


