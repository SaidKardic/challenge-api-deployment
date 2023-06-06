import pandas as pd
import numpy as np

def preprocess(json_data):
    df = pd.read_json(json_data, orient ='index')

    type_dict = {'HOUSE': 1, 'APARTMENT': 0}
    df['Type of Property'] = df.replace(type_dict, inplace=True)
