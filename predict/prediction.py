import pickle
import pandas as pd

with open('model/finalized_model.sav', 'rb') as file:
    model = pickle.load(file)

def predict(df):
    y_pred = float(model.predict(df))
    result = {'Prediction': y_pred}

    return result


