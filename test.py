import pandas as pd

UncleanData = "amazon_laptop_2023.xlsx"
CleanData = "amazon_laptop_2023_cleaned.xlsx"

# Read the data from the Excel file
df = pd.read_excel(UncleanData)
# Assuming you have a DataFrame named 'Features'
# If not, replace this with your actual DataFrame

# Create a DataFrame with unique values and their counts
unique_values_counts = df['graphics_coprocessor'].value_counts().reset_index()

# Rename the columns for clarity
unique_values_counts.columns = ['Graphics Coprocessor', 'Count']

unique_values_counts.to_csv('graphics_coprocessor_counts.txt', sep='\t', index=False)
