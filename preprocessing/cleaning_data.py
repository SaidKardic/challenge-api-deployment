import pandas as pd
import numpy as np
import re
import urllib.parse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

"""Your input is:

```json
{
  "data": {
    "area": int,
    "property-type": "APARTMENT" | "HOUSE" | "OTHERS",
    "rooms-number": int,
    "zip-code": int,
    "land-area": Optional[int],
    "garden": Optional[bool],
    "garden-area": Optional[int],
    "equipped-kitchen": Optional[bool],
    "full-address": Optional[str],
    "swimming-pool": Optional[bool],
    "furnished": Optional[bool],
    "open-fire": Optional[bool],
    "terrace": Optional[bool],
    "terrace-area": Optional[int],
    "facades-number": Optional[int],
    "building-state": Optional[
      "NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO REBUILD"
    ]
  }
}"""


def preprocess(data):
    df=pd.read_csv('new_data.csv')
    df = df[["Unnamed: 0", "Zip-code", "City", "Type of Property", 
                  "Subtype of Property", "Price", "Bedrooms","Living area",
                  "Kitchen type", "Furnished", "How many fireplaces?", "Terrace","Terrace surface",
                  "Garden","Garden surface","Surface of the plot","Number of frontages",
                  "Swimming pool", "Building condition", "Basement",
                  "Energy class", "Heating type"]]
    
    df=df.rename(columns = {'Unnamed: 0':'Property ID', 'Zip-code':'Zip code', 'Price':'Price (€)',
                             'Living area':'Living area (m²)', 'Terrace surface':'Terrace surface (m²)', 
                        'Garden surface':'Garden surface (m²)', 'Surface of the plot': 'Surface of the plot (m²)'
                          })
    
    df=df.drop_duplicates()

    columns_to_binary = ['How many fireplaces?', 'Swimming pool', 'Basement', 'Furnished']
    for column in columns_to_binary:
        df[column] = df[column].replace(np.nan,0,regex=True)
        df[column] = df[column].replace('Yes',1,regex=True)
        df[column] = df[column].replace('No',0,regex=True)

    df=df.dropna(subset=['Price (€)', 'Bedrooms', 'Living area (m²)', 'Subtype of Property'])

    df['Price (€)'] = df['Price (€)'].str.split(' ').str[-2]
    df['Price (€)'] = pd.to_numeric(df['Price (€)'])
    
    columns_to_strip = ['Living area (m²)', 'Garden surface (m²)', 'Terrace surface (m²)', 'Surface of the plot (m²)']
    for column in columns_to_strip:
        df[column] = df[column].str.split(' ').str[0]
    
    filter_G = df["Garden surface (m²)"].isnull()
    df.loc[~filter_G,'Garden'] = 'Yes'
    df.loc[df['Garden'] == 'Yes', 'Garden'] = 1
    df.loc[df["Garden"].isnull(), 'Garden'] = 0
    df.loc[df['Garden'] == 0 & df["Garden surface (m²)"].isnull(), 'Garden surface (m²)'] = 0

    filter_T = df["Terrace surface (m²)"].isnull()
    df.loc[~filter_T,'Terrace'] = 'Yes'
    df.loc[df['Terrace'] == 'Yes', 'Terrace'] = 1
    df.loc[df["Terrace"].isnull(), 'Terrace'] = 0
    df.loc[df['Terrace'] == 0 & df["Terrace surface (m²)"].isnull(), 'Terrace surface (m²)'] = 0

    df['Energy class']=df['Energy class'].replace('Not',np.nan)
    df['Energy class']=df['Energy class'].replace('Not specified',np.nan)

    df = df.astype({"Living area (m²)":"float", "Terrace":"float", "Garden":"float",
                "Terrace surface (m²)":"float",
                "Garden surface (m²)":"float","Surface of the plot (m²)":"float",
                })
    
    kitchen_type_scale={'USA hyper equipped':3, 'USA installed':2, 'USA semi equipped':1, 'USA uninstalled':0, 
                        'Hyper equipped':3, 'Installed':2, 'Semi equipped':1, 'Not installed':0}
    df['Kitchen type scale'] = df['Kitchen type'].map(kitchen_type_scale)

    building_condition_scale={'As new':6,'Just renovated':5, 'Good':4, 'To renovate':2, 
                              'To restore':1, 'To be done up':3}
    df['Building condition scale'] = df['Building condition'].map(building_condition_scale)

    def clean_city_name(city):
        # Remove quotes
        city = city.replace('"', '')
        # Decode URL encoding
        city = urllib.parse.unquote(city)
        return city

    df['City'] = df['City'].apply(clean_city_name)

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    df.drop(df[df["Zip code"].str.contains("%20")].index,inplace=True)
    df.drop(df[df['Zip code'].str.len() == 5 ].index,inplace=True)
    df['Zip code'] = pd.to_numeric(df['Zip code'])
    filter_zip = (df['Zip code'] >= 1000) & (df['Zip code'] <= 9999)
    df.drop(df.loc[~filter_zip,'Zip code'].index,inplace=True)

    filt_b = (df['Zip code'] >= 1000) & (df['Zip code'] <= 1299)
    df.loc[filt_b,'Region'] = 'Brussels capital region'
    filt_w = ((df['Zip code'] >= 1300) & (df['Zip code'] <= 1499)) | ((df['Zip code'] >=4000) & (df['Zip code'] <=7999))
    df.loc[filt_w,'Region'] = 'Walloon'
    filt_f = ((df['Zip code'] >= 1500) & (df['Zip code'] <= 3999)) | ((df['Zip code'] >=8000) & (df['Zip code'] <=9999))
    df.loc[filt_f,'Region'] = 'Flemish'

    df = df.drop(df.loc[df['Price (€)'] == 35000000].index)
    df = df.drop(df.loc[df['Living area (m²)'] == 1.0].index)

    df.loc[df['Garden surface (m²)'] == 1, 'Garden surface (m²)'] = np.nan
    df.loc[df['Terrace surface (m²)'] == 1, 'Terrace surface (m²)'] = np.nan
    df.loc[df['Surface of the plot (m²)'] == 1, 'Surface of the plot (m²)'] = np.nan
    
    df=df.drop_duplicates()

    df.to_csv("updated_cleaned_data.csv", index=False)

    #PREPROCESS

    df = pd.read_csv('updated_cleaned_data.csv')

    df_processed = df.drop(['Property ID', 'City', 'Furnished'], axis=1)
    
    s = (df_processed.dtypes == 'object')
    object_cols = list(s[s].index)
    OH_encoder = OneHotEncoder(sparse=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(df_processed[object_cols]))
    OH_cols.index = df_processed.index
    OH_cols.columns = OH_encoder.get_feature_names_out()
    df_processed = df_processed.drop(object_cols, axis=1)
    df_processed = pd.concat([df_processed, OH_cols], axis=1)

    df_processed = df_processed.drop(['Kitchen type_nan', 'Building condition_nan', 
                                  'Energy class_nan', 'Heating type_nan'], axis=1)
    

    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    imputed_columns = ['Terrace surface (m²)', 'Garden surface (m²)', 'Surface of the plot (m²)',
                                        'Number of frontages', 'Kitchen type scale', 'Building condition scale']
    imputer = imputer.fit(df_processed[imputed_columns])
    df_processed[imputed_columns] = imputer.transform(df_processed[imputed_columns])

    def subset_by_iqr(df, column, whisker_width=1.5):
        """Remove outliers from a dataframe by column, including optional whiskers, removing rows for which the column 
        value are less than Q1-1.5IQR or greater than Q3+1.5IQR.
        Args: df (`:obj:pd.DataFrame`): A pandas dataframe to subset 
        column (str): Name of the column to calculate the subset from.
        whisker_width (float): Optional, loosen the IQR filter by a factor of `whisker_width` * IQR.
        Returns:
            (`:obj:pd.DataFrame`): Filtered dataframe
        """
        # Calculate Q1, Q2 and IQR
        q1 = df[column].quantile(0.25)                 
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        # Apply filter with respect to IQR, including optional whiskers
        filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
        return df.loc[filter]

    outlier_list = ['Living area (m²)', 'Price (€)', 'Bedrooms', 'Terrace surface (m²)', 'Garden surface (m²)', 'Surface of the plot (m²)']
    for variable in outlier_list:
        df_processed = subset_by_iqr(df_processed, variable, whisker_width=1.5)                                                  
