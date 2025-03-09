import pandas as pd

# Load the features.csv file
csv_path = "audio_features.csv"
df = pd.read_csv(csv_path)

# Display first few rows
print("First 5 rows of features.csv:")
print(df.head())

# Show dataset info (column names, types, missing values)
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Display summary statistics
print("\nStatistical Summary:")
print(df.describe())
