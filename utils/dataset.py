import pandas as pd
import os

# Create directory if it doesn't exist
os.makedirs('./processed', exist_ok=True)

# Load the original dataset
print("Loading dataset...")
df = pd.read_csv('snmp_2018_1hourinterval.csv')
df = df.rename(columns={'Time': 'date'})
# Display total columns for verificaation
print(f"Original dataset has {df.shape[1]} columns")

# Get the timestamp column (first column)
timestamp_col = df.columns[0]

# Select timestamp + first 10 data columns
selected_cols = [timestamp_col] + list(df.columns[1:11])
result = df[selected_cols]

print(f"Reduced dataset has {result.shape[1]} columns")

# Save the result
output_path = './processed/snmp_first10.csv'
result.to_csv(output_path, index=False)
print(f"Saved reduced dataset to {output_path}")
