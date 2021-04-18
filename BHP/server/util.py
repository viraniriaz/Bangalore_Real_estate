import json
import pickle
import numpy as np
__locations=None
__data_columns=None
__model=None


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0]=sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index]=1

    return round(__model.predict([x])[0], 2)


def get_locations_name():
    return __locations


def load_saved_artifacts():
    print('loading the save artifacts')
    global __data_columns
    global __locations

    with open("./artifacts/column.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open('./artifacts/banglore_home_prices_models.pickle','rb') as f:
        __model = pickle.load(f)
    print('loading the model is done')


if __name__ ==  "__main__":
    load_saved_artifacts()
    print(get_locations_name())
    print(get_estimated_price('1st Phase JP nagar',1000,3,3))
