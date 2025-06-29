from scipy.stats import ttest_rel, wilcoxon, pearsonr, f_oneway
import pandas as pd
import numpy as np

def calculate_icc(data):
    """
    Calculate ICC(2,1) - Two-way random effects, single measurement, absolute agreement
    Formula: (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)
    Where MSR = Mean Square for Rows (subjects)
          MSE = Mean Square Error
          MSC = Mean Square for Columns (raters/systems)
          k = number of raters/systems
          n = number of subjects
    """
    data = np.array(data)
    n, k = data.shape
    
    # Calculate means
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    grand_mean = np.mean(data)
    
    # Calculate sum of squares
    SST = np.sum((data - grand_mean) ** 2)
    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSC = n * np.sum((col_means - grand_mean) ** 2)
    SSE = SST - SSR - SSC
    
    # Calculate mean squares
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    
    # Calculate ICC(2,1)
    if MSR + (k-1)*MSE + k*(MSC-MSE)/n == 0:
        return None, None
    
    icc = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)
    
    # Calculate F-statistic and p-value for significance test
    f_stat = MSR / MSE
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    
    # P-value from F-distribution
    from scipy.stats import f
    p_value = 1 - f.cdf(f_stat, df1, df2)
    
    return icc, p_value

# Load the CSV file
file_path = "filtered_kinematics_summary_NEW_renamed.csv"
df = pd.read_csv(file_path)

# Step 1: Identify kinematic variables
kinematic_cols = [col for col in df.columns if 'angle_' in col and 'rom' in col]

# Step 2: Filter relevant columns (including condition)
relevant_cols = ['participant_id', 'action', 'condition', 'measurement_system'] + kinematic_cols
df_filtered = df[relevant_cols].dropna()

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

# Perform paired tests per action and condition with correlation and ICC analysis
grouped_results = []

for sys1, sys2 in comparison_pairs:
    df_pair = df_subset[df_subset['measurement_system'].isin([sys1, sys2])]

    # Group by both action and condition
    for (action, condition), group in df_pair.groupby(["action", "condition"]):
        df_filtered = group[['participant_id', 'action', 'condition', 'measurement_system'] + kinematic_cols].dropna()

        pivoted = df_filtered.pivot_table(
            index='participant_id',
            columns='measurement_system',
            values=kinematic_cols
        ).dropna()

        for var in kinematic_cols:
            try:
                x = pivoted[(var, sys1)]
                y = pivoted[(var, sys2)]

                # Only test if there are at least 3 pairs (minimum for ICC)
                if len(x) < 3 or len(y) < 3:
                    continue

                # Paired t-test
                t_stat, t_p = ttest_rel(x, y)
                
                # Wilcoxon test
                try:
                    w_stat, w_p = wilcoxon(x, y)
                except ValueError:
                    w_stat, w_p = None, None

                # Pearson correlation
                try:
                    corr_coeff, corr_p = pearsonr(x, y)
                except (ValueError, RuntimeWarning):
                    corr_coeff, corr_p = None, None

                # ICC calculation
                try:
                    icc_data = np.column_stack([x, y])
                    icc_value, icc_p = calculate_icc(icc_data)
                except (ValueError, ZeroDivisionError):
                    icc_value, icc_p = None, None

                # Calculate mean and std for each system
                mean_sys1 = np.mean(x)
                std_sys1 = np.std(x, ddof=1)
                mean_sys2 = np.mean(y)
                std_sys2 = np.std(y, ddof=1)
                
                # Calculate mean difference
                mean_diff = np.mean(x - y)
                std_diff = np.std(x - y, ddof=1)

                grouped_results.append({
                    'comparison': f"{sys1} vs {sys2}",
                    'action': action,
                    'condition': condition,
                    'variable': var,
                    'n_pairs': len(x),
                    'mean_sys1': mean_sys1,
                    'std_sys1': std_sys1,
                    'mean_sys2': mean_sys2,
                    'std_sys2': std_sys2,
                    'mean_difference': mean_diff,
                    'std_difference': std_diff,
                    'paired_t_stat': t_stat,
                    'paired_t_pvalue': t_p,
                    'wilcoxon_stat': w_stat,
                    'wilcoxon_pvalue': w_p,
                    'pearson_r': corr_coeff,
                    'pearson_pvalue': corr_p,
                    'icc_value': icc_value,
                    'icc_pvalue': icc_p
                })
            except KeyError:
                continue

# Convert to DataFrame
grouped_df = pd.DataFrame(grouped_results)
grouped_df.sort_values(by='paired_t_pvalue', inplace=True)
grouped_df.reset_index(drop=True, inplace=True)

# Display results
print("Statistical Testing Results with Correlation and ICC Analysis (by Condition)")
print("=" * 80)
print(f"Total number of comparisons: {len(grouped_df)}")
print("\nFirst 10 results (sorted by t-test p-value):")
print(grouped_df[['comparison', 'action', 'condition', 'variable', 'n_pairs', 
                  'paired_t_pvalue', 'pearson_r', 'icc_value']].head(10))

# Summary statistics for correlations and ICC by condition
print(f"\nCorrelation and ICC Summary by Condition:")
condition_summary = grouped_df.groupby('condition').agg({
    'pearson_r': ['count', 'mean', 'std'],
    'icc_value': ['mean', 'std', 'min', 'max'],
    'pearson_pvalue': lambda x: (x < 0.05).sum(),
    'icc_pvalue': lambda x: (x < 0.05).sum()
}).round(3)
condition_summary.columns = ['Count', 'Mean_r', 'Std_r', 'Mean_ICC', 'Std_ICC', 'Min_ICC', 'Max_ICC', 'Sig_Corr', 'Sig_ICC']
print(condition_summary)

# ICC interpretation summary
print(f"\nICC Interpretation Guide:")
print("< 0.50: Poor reliability")
print("0.50-0.75: Moderate reliability") 
print("0.75-0.90: Good reliability")
print("> 0.90: Excellent reliability")

# Count ICC categories
icc_vals = grouped_df['icc_value'].dropna()
poor_icc = (icc_vals < 0.50).sum()
moderate_icc = ((icc_vals >= 0.50) & (icc_vals < 0.75)).sum()
good_icc = ((icc_vals >= 0.75) & (icc_vals < 0.90)).sum()
excellent_icc = (icc_vals >= 0.90).sum()

print(f"\nICC Categories:")
print(f"Poor (< 0.50): {poor_icc}")
print(f"Moderate (0.50-0.75): {moderate_icc}")
print(f"Good (0.75-0.90): {good_icc}")
print(f"Excellent (> 0.90): {excellent_icc}")

# Overall summary
print(f"\nOverall Summary:")
print(f"Mean Pearson r: {grouped_df['pearson_r'].mean():.3f}")
print(f"Mean ICC: {grouped_df['icc_value'].mean():.3f}")
print(f"Median ICC: {grouped_df['icc_value'].median():.3f}")

# Count significant results
significant_corr = grouped_df[grouped_df['pearson_pvalue'] < 0.05]
significant_icc = grouped_df[grouped_df['icc_pvalue'] < 0.05]
print(f"Significant correlations (p < 0.05): {len(significant_corr)} out of {len(grouped_df)}")
print(f"Significant ICC values (p < 0.05): {len(significant_icc)} out of {len(grouped_df)}")

# Save the results to a CSV file
output_file = "statistical_testing_results_NEW_VIVA_with_correlation_ICC_by_condition.csv"
grouped_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Summary by comparison pair and condition
print("\nICC Summary by Comparison Pair and Condition:")
icc_summary = grouped_df.groupby(['comparison', 'condition']).agg({
    'icc_value': ['mean', 'std', 'count'],
    'icc_pvalue': lambda x: (x < 0.05).sum()
}).round(3)
icc_summary.columns = ['Mean_ICC', 'Std_ICC', 'Count', 'Significant_ICC']
print(icc_summary)

# Compare conditions for ICC
print("\nICC Differences Between Conditions:")
tight_icc = grouped_df[grouped_df['condition'] == 'Tight']['icc_value'].mean()
loose_icc = grouped_df[grouped_df['condition'] == 'Loose']['icc_value'].mean()
print(f"Average ICC - Tight condition: {tight_icc:.3f}")
print(f"Average ICC - Loose condition: {loose_icc:.3f}")
print(f"Difference (Tight - Loose): {tight_icc - loose_icc:.3f}")

# Best performing measurement pairs
print("\nTop 5 Measurement Pairs by ICC:")
top_icc = grouped_df.nlargest(5, 'icc_value')[['comparison', 'action', 'condition', 'variable', 'icc_value', 'icc_pvalue']]
print(top_icc)