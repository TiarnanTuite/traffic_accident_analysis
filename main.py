import pandas as pd

# Loading cleaned dataset
df = pd.read_csv(
    "C:/Users/tuite/Desktop/Software Portfolio/python/Traffic_Accident_Analysis/data/cleaned_traffic/accidents.csv"
)

# Checking the first few rows to inspect the dataset
print(df.head())
