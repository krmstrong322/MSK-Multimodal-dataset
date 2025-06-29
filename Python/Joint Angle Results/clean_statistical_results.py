import pandas as pd

file = "statistical_testing_results_NEW_VIVA.csv"

# Load the results
df = pd.read_csv(file)

# Filter out rows containing strings with 'ankle' in them
df_filtered = df[~df['variable'].str.contains('ankle', case=False, na=False)]
# Filter out rows containing 'balance' in the 'action' column
df_filtered = df_filtered[~df_filtered['action'].str.contains('balance', case=False, na=False)]
# Filter out rows containing 'lunge' in the 'action' column

df_filtered = df_filtered[~df_filtered['action'].str.contains('lunge', case=False, na=False)]

# Save the filtered results to a new CSV file
df_filtered.to_csv("filtered_statistical_testing_results_NEW_VIVA.csv", index=False)
# This script filters out rows from the statistical testing results that contain 'ankle' in the 'variable' column.

