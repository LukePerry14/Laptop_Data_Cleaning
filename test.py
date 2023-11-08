import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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
                return int(vals[0]) * 1024
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

def process_cpu(input):
    allowed_cpus = ['core i3', 'core i5', 'core i7', 'core i9', 'ryzen 3', 'ryzen 5', 'ryzen 7', 'ryzen 9', 'celeron', 'ryzen r series']
    result = process.extractOne(str(input), allowed_cpus)
    value, score = result
    return value

def process_os(input):
    allowed_os = ['windows', 'mac os', 'chrome os']
    result = process.extractOne(str(input), allowed_os)
    value, score = result
    return value

def combine_graphics(Features):
    graphics = str(Features['graphics'])
    if graphics != 'integrated':
        return str(Features['graphics_coprocessor'])
    else:
        return graphics

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

df['model'] = df.apply(concatenate_brand_model, axis=1)
df = df.drop_duplicates(subset=['model'])
df = df.drop('brand', axis=1)

#fill in missing values in special_features and color
df['special_features'] = df['special_features'].fillna('none')
df['color'] = df['color'].fillna('unknown')


#change os to be one of the three main ones by fuzzy string matching
df['OS'] = df['OS'].apply(process_os)


#combine graphics and graphics_coprocessor columns, where laptops with non integrated graphics have the card specified in the graphics column
df['graphics'] = df.apply(combine_graphics, axis=1)
df = df.drop('graphics_coprocessor', axis=1)

print(df)

