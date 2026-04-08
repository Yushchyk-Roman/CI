import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import INDENT


def data_cleaning(raw_data):
    mapping_days = {
        "Weekend": 0,
        "Weekday": 1
    }
    raw_data['Day of Week'] = raw_data['Day of Week'].map(mapping_days)

    raw_data = pd.get_dummies(raw_data, columns=['Building Type'], drop_first=True, dtype=int) 

    target_col = 'Energy Consumption'

    x_values = raw_data.drop(columns=target_col)
    y_values = raw_data[target_col]
    
    return x_values, y_values

def data_report(raw_data):
    print(f"{INDENT} Example of data:")
    print(raw_data.head())

    print(f"{INDENT} Core information about data:")
    raw_data.info()
    print(f"{INDENT} Analysis main info about data:")
    print(raw_data.describe())

    print(f"{INDENT} Checking if data have NULL values:")
    print(raw_data.isnull().sum())

def preparetion(PATH_TO_TRAIN_CSV, PATH_TO_TEST_CSV, need_report=False):
    train_raw = pd.read_csv(PATH_TO_TRAIN_CSV)
    test_raw = pd.read_csv(PATH_TO_TEST_CSV)
    
    if need_report:
        data_report(train_raw)
    
    x_train, y_train = data_cleaning(train_raw)
    x_test, y_test = data_cleaning(test_raw)

    scaler = StandardScaler()
    num_colums = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']

    x_train[num_colums] = scaler.fit_transform(x_train[num_colums] )
    x_test[num_colums] = scaler.transform(x_test[num_colums])
    
    return x_train, x_test, y_train, y_test