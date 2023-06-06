import pandas as pd

def preprocess(json_data):
    df = pd.DataFrame(json_data, index=[0])

    df = df.rename(columns = {'zip_code':'Zip code', 'property_type':'Type of Property', 'area':'Living area (m²)', 
                              'rooms_number':'Bedrooms', 'garden':'Garden', 'terrace':'Terrace', 'terrace_area':'Terrace surface (m²)', 
                        'garden_area':'Garden surface (m²)', 'equipped_kitchen':'Kitchen type scale', 'swimming_pool':'Swimming pool', 
                        'furnished':'Furnished', 'open_fire':'How many fireplaces?', 'land_area': 'Surface of the plot (m²)', 
                        'facades_number':'Number of frontages', 'building_state': 'Building condition scale'})
    
    property_type_dict = {'HOUSE': 1, 'APARTMENT': 0}
    df['Type of Property'] = df.replace(property_type_dict, inplace=True)

    boolean_dict = {False: 0, True: 1}
    columns_to_binary = ['Kitchen type scale', 'Swimming pool', 'Garden', 'Terrace', 'Furnished', 'How many fireplaces?']
    for column in columns_to_binary:
        df[column].replace(boolean_dict, inplace=True)

    building_state_dict = {'NEW': 5, 'JUST RENOVATED': 4,
                     'GOOD': 3, 'TO RENOVATE': 2, 'TO REBUILD': 1}
    df['Building condition scale'].replace(building_state_dict, inplace=True)

    columns_missing_values = ['Kitchen type scale', 'Swimming pool', 'Garden', 'Garden surface (m²)', 'Terrace', 
                              'Terrace surface (m²)', 'Furnished', 'How many fireplaces?']
    for column in columns_missing_values:
        df[column].fillna(0, inplace=True)
    
    df['Number of frontages'].fillna(1, inplace=True)

    filt_b = (df['Zip code'] >= 1000) & (df['Zip code'] <= 1299)
    df.loc[filt_b,'Region'] = 2
    filt_w = ((df['Zip code'] >= 1300) & (df['Zip code'] <= 1499)) | ((df['Zip code'] >=4000) & (df['Zip code'] <=7999))
    df.loc[filt_w,'Region'] = 1
    filt_f = ((df['Zip code'] >= 1500) & (df['Zip code'] <= 3999)) | ((df['Zip code'] >=8000) & (df['Zip code'] <=9999))
    df.loc[filt_f,'Region'] = 3

    df = df[['Zip code', 'Type of Property', 'Bedrooms',
       'Living area (m²)', 'Furnished', 'How many fireplaces?', 'Terrace',
       'Terrace surface (m²)', 'Garden', 'Garden surface (m²)',
       'Surface of the plot (m²)', 'Number of frontages', 'Swimming pool',
       'Kitchen type scale', 'Building condition scale', 'Region']]

    return df

