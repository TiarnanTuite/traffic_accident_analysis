import pandas as pd
import os

# Path to your big dataset
input_path = os.path.join("data", "final_cleaned_accident_data.csv")

# Load full dataset
print("Loading full dataset...")
df = pd.read_csv(input_path)
print(f"Full dataset shape: {df.shape}")
con
# Take a random sample of 10,000 rows (adjust as needed)
sample_size = 10000
df_sample = df.sample(sample_size, random_state=42)

# Save trimmed dataset
output_path = os.path.join("data", "sample_accident_data.csv")
df_sample.to_csv(output_path, index=False)

print(f"Saved smaller dataset: {output_path}")
print(f"Trimmed dataset shape: {df_sample.shape}")
