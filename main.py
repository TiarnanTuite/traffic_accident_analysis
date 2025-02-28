import pandas as pd

# Loading cleaned dataset
df = pd.read_csv(
    "C:/Users/tuite/Desktop/Software Portfolio/python/Traffic_Accident_Analysis/data/final_cleaned_accident_data.csv"
)

# Checking the first few rows to inspect the dataset
print(df.head())
