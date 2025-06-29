from scipy.stats import ttest_rel, wilcoxon

import pandas as pd

# Load the CSV file
file_path = "kinematics_summary_NEW.csv"
df = pd.read_csv(file_path)

# Step 1: Identify kinematic variables
kinematic_cols = [col for col in df.columns if 'angle_' in col and 'rom' in col]

# Step 2: Filter relevant columns
relevant_cols = ['participant_id', 'action', 'measurement_system'] + kinematic_cols
df_filtered = df[relevant_cols].dropna()

# Step 3: Pivot to align measurement systems (assuming exactly 2 per participant-action)
pivoted = df_filtered.pivot_table(
    index=['participant_id', 'action'],
    columns='measurement_system',
    values=kinematic_cols
)

# Remove rows where any system is missing
pivoted = pivoted.dropna()

# # Step 4: Perform paired t-test and Wilcoxon for each kinematic variable
# results = []
# systems = df['measurement_system'].unique()
# if len(systems) != 2:
#     raise ValueError(f"Expected exactly 2 measurement systems, found: {systems}")

# sys1, sys2 = systems

# for var in kinematic_cols:
#     x = pivoted[(var, sys1)]
#     y = pivoted[(var, sys2)]

#     # Paired t-test
#     t_stat, t_p = ttest_rel(x, y)

#     # Wilcoxon test (only works if no ties or zero-differences)
#     try:
#         w_stat, w_p = wilcoxon(x, y)
#     except ValueError:
#         w_stat, w_p = None, None

#     results.append({
#         'variable': var,
#         'paired_t_stat': t_stat,
#         'paired_t_pvalue': t_p,
#         'wilcoxon_stat': w_stat,
#         'wilcoxon_pvalue': w_p
#     })

# # Convert to DataFrame for display
# results_df = pd.DataFrame(results)
# results_df.sort_values(by='paired_t_pvalue')

# Define the measurement system pairs to compare
# comparison_pairs = [
#     ('IMU', 'Mediapipe1'),
#     ('IMU', 'MoCap'),
#     ('IMU', 'VIBE1'),
#     ('MoCap', 'Mediapipe1'),
#     ('MoCap', 'VIBE1'),
# ]

# # Prepare results container
# all_results = []

# # Filter for only relevant measurement systems
# df_subset = df[df['measurement_system'].isin(set(sum(comparison_pairs, ())))]

# # Iterate through each comparison pair
# for sys1, sys2 in comparison_pairs:
#     # Filter and pivot data for the current pair
#     df_pair = df_subset[df_subset['measurement_system'].isin([sys1, sys2])]
#     df_filtered = df_pair[['participant_id', 'action', 'measurement_system'] + kinematic_cols].dropna()
    
#     pivoted = df_filtered.pivot_table(
#         index=['participant_id', 'action'],
#         columns='measurement_system',
#         values=kinematic_cols
#     ).dropna()

#     for var in kinematic_cols:
#         try:
#             x = pivoted[(var, sys1)]
#             y = pivoted[(var, sys2)]

#             t_stat, t_p = ttest_rel(x, y)
#             try:
#                 w_stat, w_p = wilcoxon(x, y)
#             except ValueError:
#                 w_stat, w_p = None, None

#             all_results.append({
#                 'comparison': f"{sys1} vs {sys2}",
#                 'variable': var,
#                 'paired_t_stat': t_stat,
#                 'paired_t_pvalue': t_p,
#                 'wilcoxon_stat': w_stat,
#                 'wilcoxon_pvalue': w_p
#             })
#         except KeyError:
#             continue

# # Convert results to DataFrame and display
# results_df = pd.DataFrame(all_results)
# results_df.sort_values(by='paired_t_pvalue', inplace=True)
# results_df.reset_index(drop=True, inplace=True)
# print(results_df.head(10))

kinematic_cols = [col for col in df.columns if 'angle_' in col and 'rom' in col]

# Re-define system pairs
comparison_pairs = [
    ('IMU', 'Mediapipe1'),
    ('IMU', 'Mediapipe2'),
    ('IMU', 'MoCap'),
    ('IMU', 'VIBE1'),
    ('IMU', 'VIBE2'),
    ('MoCap', 'Mediapipe1'),
    ('MoCap', 'Mediapipe2'),
    ('MoCap', 'VIBE1'),
    ('MoCap', 'VIBE2'),
    ('Mediapipe1', 'VIBE1'),
    ('Mediapipe2', 'VIBE2'),

]

# Filter data
df_subset = df[df['measurement_system'].isin(set(sum(comparison_pairs, ())))]

# Perform paired tests per action
grouped_results = []

for sys1, sys2 in comparison_pairs:
    df_pair = df_subset[df_subset['measurement_system'].isin([sys1, sys2])]

    for action, group in df_pair.groupby("action"):
        df_filtered = group[['participant_id', 'action', 'measurement_system'] + kinematic_cols].dropna()

        pivoted = df_filtered.pivot_table(
            index='participant_id',
            columns='measurement_system',
            values=kinematic_cols
        ).dropna()

        for var in kinematic_cols:
            try:
                x = pivoted[(var, sys1)]
                y = pivoted[(var, sys2)]

                # Only test if there are at least 2 pairs
                if len(x) < 2 or len(y) < 2:
                    continue

                t_stat, t_p = ttest_rel(x, y)
                try:
                    w_stat, w_p = wilcoxon(x, y)
                except ValueError:
                    w_stat, w_p = None, None

                grouped_results.append({
                    'comparison': f"{sys1} vs {sys2}",
                    'action': action,
                    'variable': var,
                    'n_pairs': len(x),
                    'paired_t_stat': t_stat,
                    'paired_t_pvalue': t_p,
                    'wilcoxon_stat': w_stat,
                    'wilcoxon_pvalue': w_p
                })
            except KeyError:
                continue

# Convert to DataFrame
grouped_df = pd.DataFrame(grouped_results)
grouped_df.sort_values(by='paired_t_pvalue', inplace=True)
grouped_df.reset_index(drop=True, inplace=True)
print(grouped_df)
# Save the results to a CSV file
grouped_df.to_csv("statistical_testing_results_NEW_VIVA.csv", index=False)