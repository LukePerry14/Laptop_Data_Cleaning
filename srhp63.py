import pandas as pd
import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


from fuzzywuzzy import process

UncleanData = "amazon_laptop_2023.xlsx"
CleanData = "amazon_laptop_2023_cleaned.xlsx"
# Read the data from the Excel file
df = pd.read_excel(UncleanData)


# def get_mode(series):
#     non_nan_series = series.dropna()
#     mode_series = non_nan_series.mode()

#     if mode_series.empty:
#         return pd.NA
    
#     return mode_series.iloc[0]

def concatenate_brand_model(Features):
    brand = str(Features['brand'])
    model = str(Features['model'])
    if brand not in model:
        return brand + ' ' + model
    else:
        return model 
    
def process_storage(value):
    if type(value) == str:
        vals = value.split(' ')
        vals[0] = float(vals[0])
        if len(vals) == 2:
            if vals[1] == 'tb':
                return int(vals[0]) * 1000
        return str(round(vals[0]))
    return str(value)

def price_regression(df, target):
    X = df.dropna(subset=[target])[['price']]
    y = df.dropna(subset=[target])[target]

    tree_model = DecisionTreeRegressor()
    tree_model.fit(X, y)
    X_missing = df[df[target].isna()][['price']]
    predicted_values = tree_model.predict(X_missing)
    return predicted_values

def rating_regression(df):
    test = df.copy()
    test = test.dropna(subset=['rating'])

    X = test[['screen_size', 'harddisk', 'ram', 'OS', 'graphics']]
    y = test['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['OS', 'graphics'])
        ],
        remainder='passthrough'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Training the Decision Tree Regressor
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(X_train_encoded, y_train)

    # Making predictions
    y_pred = decision_tree_model.predict(X_test_encoded)
    y_pred_rescaled = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred)) * 5

    mse = mean_squared_error(y_test, y_pred_rescaled)
    r2 = r2_score(y_test, y_pred_rescaled)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

def actual_price_regression(df):
    test = df.copy()
    test = test.dropna(subset=['price'])

    X = test[['screen_size', 'harddisk', 'ram', 'OS', 'graphics']]
    y = test['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['OS', 'graphics'])
        ],
        remainder='passthrough'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Training the Decision Tree Regressor
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(X_train_encoded, y_train)

    # Making predictions
    y_pred = decision_tree_model.predict(X_test_encoded)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

def merge_model(cols):
    brand = str(cols['brand'])
    model = str(cols['model'])
    if brand not in model:
        return brand + ' ' + model
    else:
        return model

def process_color(input):
    allowed_colors = ['grey', 'silver', 'blue', 'gold', 'black', 'unknown', 'green', 'pink', 'red']
    result = process.extractOne(str(input), allowed_colors)
    value, score = result
    return value

def cpu_shrink(cpu_column):
    allowed_cpus = ['core i3', 'core i5', 'core i7', 'core i9', 'ryzen 3', 'ryzen 5', 'ryzen 7', 'ryzen 9', 'celeron', 'ryzen r series']

    for i in range(len(cpu_column)):
        value = cpu_column.iloc[i]
        if (value == "unknown") or (value == "nan") or (value == "others"):
            cpu_column.iat[i] = "nan"
            continue
        result = process.extractOne(str(value), allowed_cpus)
        match_value, score = result
        if score < 90:
            allowed_cpus.append(value)
            cpu_column.iat[i] = value
        else:
            cpu_column.iat[i] = match_value

    return cpu_column

def process_os(input):
    allowed_os = ['windows 7', 'windows 8', 'windows 10 home', 'windows 10 pro', 'windows 11 home', 'windows 11 pro', 'mac os', 'chrome os']
    result = process.extractOne(str(input), allowed_os)
    value, score = result
    return value

def combine_graphics(Features):
    coprocessor = str(Features['graphics_coprocessor'])
    graphics = str(Features['graphics'])
    if coprocessor != "nan":
        if (coprocessor == "intel") or (coprocessor == "embedded"):
            return 'integrated'
        return coprocessor
    elif (graphics != "nan"):
        return graphics
    else:
        return "integrated"

def graphics_shrink(graphics_column):
    processors = ['integrated']

    for i in range(len(graphics_column)):
        value = graphics_column.iloc[i]
        result = process.extractOne(str(value), processors)
        match_value, score = result
        if score < 90:
            processors.append(value)
            graphics_column.iat[i] = value
        else:
            graphics_column.iat[i] = match_value

    return graphics_column
    
def round_to_power_of_2(x):
    #formula given by user Paul Dixon on link https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
    return pow(2, math.ceil(math.log(x) / math.log(2)))


#remove rows without specified model
df = df.dropna(subset=['model'])

#delete cpu_speed column
df = df.drop('cpu_speed', axis=1)

#standardise all colors into one of a few options with fuzzy string matching.
df['color'] = df['color'].apply(process_color)

#convert price to float and remove $ sign
df['price'] = df['price'].str.replace(r'[^0-9.]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

#remove whitespace and decapitalise all strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()
        df[col] = df[col].str.lower()

#convert all storage to gb, remove strings from screen_size, ram, and harddisk, convert all to floats and fill unknowns with predictions from scikit decision tree regression model.
df['harddisk'] = df['harddisk'].apply(process_storage)
for col in ['screen_size', 'ram', 'harddisk']:
    df[col] = df[col].str.replace(r'[^0-9.]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df.loc[df[col].isna(), col] = price_regression(df, col)

#round all screen sizes to 1 decimal place, then round all ram and harddisk values to nearest power of two in accordance with hard disk standards
df['screen_size'] = df['screen_size'].round(1)
df['ram'] = df['ram'].apply(round_to_power_of_2)
df['harddisk'] = df['harddisk'].apply(round_to_power_of_2)


#emove model duplicates, then merge the brand and model columns
df = df.drop_duplicates(subset=['model'])
df['model'] = df.apply(concatenate_brand_model, axis=1)
df = df.drop('brand', axis=1)

#fill in missing values in special_features and color
df['special_features'] = df['special_features'].fillna('none')


#change os to be one of the three main ones by fuzzy string matching
df['OS'] = df['OS'].apply(process_os)

#use custom string matching algorithm to shrink the number of distinct values by standardising similar entries
df['cpu'] = cpu_shrink(df['cpu'])

#combine graphics and graphics_coprocessor columns, where laptops with non integrated graphics have the card specified in the graphics column
df['graphics'] = df.apply(combine_graphics, axis=1)

#use custom string matching algorithm to shrink the number of distinct values by standardising similar entries
df['graphics'] = graphics_shrink(df['graphics'])

df = df.drop('graphics_coprocessor', axis=1)

rating_regression(df)
#print(df)

