import pandas as pd

# Read correlation values from the text file
with open('sorted_corr_values2.txt', 'r') as file:
    corr_values = file.readlines()

# Create a dictionary to store column correlations
column_correlations = {}

# Parse correlation values and populate the dictionary
for line in corr_values:
    line = line.strip()
    if ':' in line:
        column, correlation = line.rsplit(':', 1)
        column_correlations[column] = float(correlation)

# Convert the dictionary to a pandas Series
correlation_series = pd.Series(column_correlations)

# Identify columns with correlation less than 0.2
columns_to_drop = correlation_series[correlation_series < 0.2].index


df = pd.read_csv('kc_house_cleaned.csv')
# Drop the columns from the DataFrame
df = df.drop(columns_to_drop, axis=1)
df['date'] = df['date'].str.replace('T.*', '', regex=True)
df.to_csv('kc_house_High_Corr.csv',index=False)
