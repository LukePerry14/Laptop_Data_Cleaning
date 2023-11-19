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
    #establish columns
    brand = str(Features['brand'])
    model = str(Features['model'])
    #check brand isn't already in model
    if brand not in model:
        return brand + ' ' + model
    else:
        return model 
    
def process_storage(value):
    #check if value contains unit
    if type(value) == str:

        #remove unit
        vals = value.split(' ')
        vals[0] = float(vals[0])

        #check if value needs converting to gb and return converted value
        if (vals[1] == 'tb'):
            return int(vals[0]) * 1000
        
        #if no conversion needed return value
        return str(round(vals[0]))
    return str(value)

def price_regression(df, target):

    #select datapoints with values for price to train regression model
    X = df.dropna(subset=[target])[['price']]
    y = df.dropna(subset=[target])[target]

    #train model
    tree_model = DecisionTreeRegressor()
    tree_model.fit(X, y)

    #identify target records for regression
    X_missing = df[df[target].isna()][['price']]

    #predict values
    predicted_values = tree_model.predict(X_missing)
    return predicted_values

def rating_regression_test(df):
    #function to test the use of a regression model for the prediction of the rating values by evaluating the MSE and R-sqaured value

    #take copies of df to ensure original values unchanged
    test = df.copy()
    test = test.dropna(subset=['rating'])

    #select columns for data
    X = test[['screen_size', 'harddisk', 'ram', 'OS', 'graphics']]
    y = test['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #preprocess categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['OS', 'graphics'])
        ],
        remainder='passthrough'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    #train model with data
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(X_train_encoded, y_train)

    #make predictions and evaluate performance
    y_pred = decision_tree_model.predict(X_test_encoded)
    y_pred_rescaled = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred)) * 5

    mse = mean_squared_error(y_test, y_pred_rescaled)
    r2 = r2_score(y_test, y_pred_rescaled)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

def price_regression_test(df):
    #function to test the use of a regression model for the prediction of the price values by evaluating the MSE and R-sqaured value

    #take copies of df to ensure original values unchanged
    test = df.copy()
    test = test.dropna(subset=['price'])

    #select columns for data
    X = test[['screen_size', 'harddisk', 'ram', 'OS', 'graphics']]
    y = test['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #preprocess categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['OS', 'graphics'])
        ],
        remainder='passthrough'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    #train model with data
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(X_train_encoded, y_train)

    #make predictions and evaluate performance
    y_pred = decision_tree_model.predict(X_test_encoded)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

def full_price_regression(df):

    #filter df to get training data for regression model and prediction data
    train_data = df.dropna(subset=['price'])
    predict_data = df[df['price'].isna()]
    X_train = train_data[['screen_size', 'harddisk', 'ram', 'OS', 'graphics']]
    y_train = train_data['price']
    X_predict = predict_data[['screen_size', 'harddisk', 'ram', 'OS', 'graphics']]

    #preprocess categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['OS', 'graphics'])
        ],
        remainder='passthrough'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_predict_encoded = preprocessor.transform(X_predict)

    #train model with data
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(X_train_encoded, y_train)

    #make predictions for records without prices rounded to 2dp
    predicted_prices = decision_tree_model.predict(X_predict_encoded).round(2)

    #reduce all predicted values < 650 above 1500 to 1500 (accounting for outlier values that will be disregarded due to price in Q2 regardless of unknown price - as MSE calculated as 389086 in price_regression_test)
    predicted_prices[predicted_prices < 2150] = 1500

    #reinsert data into df
    df.loc[df['price'].isna(), 'price'] = predicted_prices

    return df

def process_color(input):
    #enforce all values in color column take one of the below colors
    allowed_colors = ['grey', 'silver', 'blue', 'gold', 'black', 'unknown', 'green', 'pink', 'red']

    #run fuzzy string matching on 'color' column
    result = process.extractOne(str(input), allowed_colors)
    value, score = result
    return value

def cpu_shrink(cpu_column):
    #initial set of allowed values
    allowed_cpus = ['core i3', 'core i5', 'core i7', 'core i9', 'ryzen 3', 'ryzen 5', 'ryzen 7', 'ryzen 9', 'celeron', 'ryzen r series']


    for i in range(len(cpu_column)):
        value = cpu_column.iloc[i]

        #maintain unknown values
        if (value == "unknown") or (value == "nan") or (value == "others"):
            cpu_column.iat[i] = "nan"
            continue
        
        #run fuzzy string matching on value
        result = process.extractOne(str(value), allowed_cpus)
        match_value, score = result

        #if matching confidence less than 90% insert value into set of allowed cpus, otherwise use matched value.
        if score < 90:
            allowed_cpus.append(value)
            cpu_column.iat[i] = value
        else:
            cpu_column.iat[i] = match_value

    #return shrunk column
    return cpu_column

def os_shrink(os_column):
    #initial set of allowed values
    allowed_os = ['windows 7', 'windows 8', 'windows 10 home', 'windows 10 pro', 'windows 11 home', 'windows 11 pro', 'mac os', 'chrome os']


    for i in range(len(os_column)):
        value = os_column.iloc[i]

        #maintain unknown values
        if (value == "unknown") or (value == "nan") or (value == "others"):
            os_column.iat[i] = "nan"
            continue

        #run fuzzy string matching on value
        result = process.extractOne(str(value), allowed_os)
        match_value, score = result

        #if matching confidence less than 90% insert value into set of allowed values, otherwise use matched value.
        if score < 90:
            allowed_os.append(value)
            os_column.iat[i] = value
        else:
            os_column.iat[i] = match_value

    #return shrunk column
    return os_column

def special_shrink(special_column):
    #initial set of allowed values
    allowed_special = ['wifi', 'bluetooth', 'anti glare', 'fingerprint reader', 'backlit keyboard', 'hd audio', 'stylus', 'security slot', 'memory card slot','bezel', 'corning gorilla glass']


    for i in range(len(special_column)):
        inp = []

        #leave unknown values as an empty set of features
        if (str(special_column.iloc[i]) == "nan"):
            special_column.iat[i] = []
            continue

        #convert string into array of features
        values = special_column.iloc[i].split(',')

        for value in values:
            value = value.strip()

            #handle obvious pitfalls i.e. n/a values, sentences which contain commas, and empty values
            if (value == "information not available"):
                special_column.iat[i] = []
                break
            elif (value.count(" ") > 3) or (value == ""):
                continue
            
            #run fuzzy string matching on value
            result = process.extractOne(str(value), allowed_special)
            match_value, score = result

            #if matching confidence less than 90% insert value into set of allowed values(unless one of the major outliers that already matches a value in the allowed array), otherwise use matched value and add to the features array for that record.
            if score < 90:
                if(value == "backlit kb"):
                    inp.append("backlit keyboard")
                elif(value == "fingerprint sensor"):
                    inp.append("fingerprint reader")
                elif(value == 'pen'):
                    inp.append("stylus")
                else:
                    allowed_special.append(value)
                    inp.append(value)
            else:
                inp.append(match_value)

        #set value for that record as array of identified features
        special_column.iat[i] = inp
    
    #return shrunk and reformated column
    return special_column

def special_clean(special_column):
    #function to remove features which have cardinality one and are therefore outliers

    values = {}

    for i in range(len(special_column)):

        for feature in special_column.iloc[i]:
            #add index of record to associated feature in dictionary
            if feature in values:
                values[feature].append(i)
            else:
                values[feature] = [i]
    
    #remove any features that only appear once from the record that they appear
    for x in values.keys():
        if len(values[x]) == 1:
            special_column.iloc[values[x][0]].remove(x)

    #return cleaned column
    return special_column

def combine_graphics(Features):
    #define columns
    coprocessor = str(Features['graphics_coprocessor'])
    graphics = str(Features['graphics'])

    if coprocessor != "nan":

        #check whether coprocessor column incorrectly used to declare integrated graphics
        if (coprocessor == "intel") or (coprocessor == "embedded"):
            return 'integrated'
        
        #otherwise use dedicated card
        return coprocessor
    
    #return the integrated graphics declared in the 'graphics' column or assume laptop uses integrated graphics
    elif (graphics != "nan"):
        return graphics
    else:
        return "integrated"

def graphics_shrink(graphics_column):
    #initial set of allowed values
    processors = ['integrated']

    for i in range(len(graphics_column)):
        value = graphics_column.iloc[i]

        #run fuzzy string matching on value
        result = process.extractOne(str(value), processors)
        match_value, score = result

        #if matching confidence less than 90% insert value into set of allowed cpus, otherwise use matched value.
        if score < 90:
            processors.append(value)
            graphics_column.iat[i] = value
        else:
            graphics_column.iat[i] = match_value

    #return shrunk column
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

#use custom string matching algorithm to shrink the number of distinct values by standardising similar entries
df['cpu'] = cpu_shrink(df['cpu'])

#shrink number of possible OS values with fuzzy string matching
df['OS'] = os_shrink(df['OS'])

#change os to be one of the three main ones by fuzzy string matching
df['special_features'] = df['special_features'].str.replace('&',',')
df['special_features'] = special_shrink(df['special_features'])
df['special_features'] = special_clean(df['special_features'])

#combine graphics and graphics_coprocessor columns, where laptops with non integrated graphics have the card specified in the graphics column
df['graphics'] = df.apply(combine_graphics, axis=1)

#use custom string matching algorithm to shrink the number of distinct values by standardising similar entries
df['graphics'] = graphics_shrink(df['graphics'])

df = df.drop('graphics_coprocessor', axis=1)

# find mean of ratings column and fill all NaN values with this value
mean = df['rating'].mean().round(1)
print(mean)
df['rating'] = df['rating'].fillna(mean)

#remove all records with rating value less than the mean
df = df[df['rating'] >= mean]

#use decision tree regression to predict values for records with NaN for price
df = full_price_regression(df)

#drop all records with price greater than 1500 for Q2
df = df[df['price'] <= 1500]

df.to_excel(CleanData, index=False)
print(df)

