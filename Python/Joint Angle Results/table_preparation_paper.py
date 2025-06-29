import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

def separate_by_modality_comparison(
    df: pd.DataFrame, 
    comparison_col: str = 'comparison',
    output_dir: Optional[str] = None,
    save_individual_files: bool = False,
    significance_level: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Separate statistical testing results by modality comparison.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing statistical testing results
    comparison_col : str
        Column name containing the comparison information (default: 'comparison')
    output_dir : str, optional
        Directory to save individual CSV files for each comparison
    save_individual_files : bool
        Whether to save individual CSV files for each comparison
    significance_level : float
        Significance threshold for marking results (default: 0.05)
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with comparison names as keys and filtered DataFrames as values
    """
    
    # Validate input
    if comparison_col not in df.columns:
        raise ValueError(f"Column '{comparison_col}' not found in DataFrame")
    
    # Get unique comparisons
    unique_comparisons = df[comparison_col].unique()
    
    # Dictionary to store separated DataFrames
    comparison_dfs = {}
    
    # Create output directory if specified
    if output_dir and save_individual_files:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(unique_comparisons)} unique comparisons:")
    
    for comparison in unique_comparisons:
        # Filter data for this comparison
        comparison_df = df[df[comparison_col] == comparison].copy()
        
        # Add significance indicators if p-value columns exist
        if 'paired_t_pvalue' in comparison_df.columns:
            comparison_df['t_test_significant'] = comparison_df['paired_t_pvalue'] < significance_level
            comparison_df['t_test_stars'] = comparison_df['paired_t_pvalue'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            )
        
        if 'wilcoxon_pvalue' in comparison_df.columns:
            comparison_df['wilcoxon_significant'] = comparison_df['wilcoxon_pvalue'] < significance_level
            comparison_df['wilcoxon_stars'] = comparison_df['wilcoxon_pvalue'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            )
        
        # Sort by action and variable for better organization
        if 'action' in comparison_df.columns and 'variable' in comparison_df.columns:
            comparison_df = comparison_df.sort_values(['action', 'variable'])
        
        # Store in dictionary
        comparison_dfs[comparison] = comparison_df
        
        # Print summary
        n_tests = len(comparison_df)
        n_significant_t = comparison_df.get('t_test_significant', pd.Series()).sum()
        n_significant_w = comparison_df.get('wilcoxon_significant', pd.Series()).sum()
        
        print(f"  {comparison}: {n_tests} tests")
        if 'paired_t_pvalue' in comparison_df.columns:
            print(f"    - T-test significant: {n_significant_t}/{n_tests}")
        if 'wilcoxon_pvalue' in comparison_df.columns:
            print(f"    - Wilcoxon significant: {n_significant_w}/{n_tests}")
        
        # Save individual file if requested
        if save_individual_files and output_dir:
            # Clean filename (remove special characters)
            safe_filename = comparison.replace(' ', '_').replace('vs', 'vs').replace('/', '_')
            filepath = os.path.join(output_dir, f"{safe_filename}_results.csv")
            comparison_df.to_csv(filepath, index=False)
            print(f"    - Saved to: {filepath}")
    
    return comparison_dfs

def create_summary_table(comparison_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary table showing overview statistics for each comparison.
    
    Parameters:
    -----------
    comparison_dfs : Dict[str, pd.DataFrame]
        Dictionary of comparison DataFrames from separate_by_modality_comparison()
    
    Returns:
    --------
    pd.DataFrame
        Summary table with statistics for each comparison
    """
    
    summary_data = []
    
    for comparison, df in comparison_dfs.items():
        summary_row = {
            'comparison': comparison,
            'total_tests': len(df),
            'unique_actions': df['action'].nunique() if 'action' in df.columns else 0,
            'unique_variables': df['variable'].nunique() if 'variable' in df.columns else 0,
            'mean_n_pairs': df['n_pairs'].mean() if 'n_pairs' in df.columns else 0,
        }
        
        # Add significance statistics
        if 't_test_significant' in df.columns:
            summary_row['t_test_significant_count'] = df['t_test_significant'].sum()
            summary_row['t_test_significant_pct'] = (df['t_test_significant'].sum() / len(df)) * 100
        
        if 'wilcoxon_significant' in df.columns:
            summary_row['wilcoxon_significant_count'] = df['wilcoxon_significant'].sum()
            summary_row['wilcoxon_significant_pct'] = (df['wilcoxon_significant'].sum() / len(df)) * 100
        
        # Check agreement between tests
        if 't_test_significant' in df.columns and 'wilcoxon_significant' in df.columns:
            agreement = (df['t_test_significant'] == df['wilcoxon_significant']).sum()
            summary_row['test_agreement_count'] = agreement
            summary_row['test_agreement_pct'] = (agreement / len(df)) * 100
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)

def filter_significant_results(
    comparison_dfs: Dict[str, pd.DataFrame], 
    test_type: str = 'both',
    significance_level: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Filter to show only significant results from each comparison.
    
    Parameters:
    -----------
    comparison_dfs : Dict[str, pd.DataFrame]
        Dictionary of comparison DataFrames
    test_type : str
        Which test to use for filtering: 't_test', 'wilcoxon', or 'both' (default)
    significance_level : float
        Significance threshold (default: 0.05)
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with only significant results for each comparison
    """
    
    significant_dfs = {}
    
    for comparison, df in comparison_dfs.items():
        if test_type == 't_test' and 'paired_t_pvalue' in df.columns:
            mask = df['paired_t_pvalue'] < significance_level
        elif test_type == 'wilcoxon' and 'wilcoxon_pvalue' in df.columns:
            mask = df['wilcoxon_pvalue'] < significance_level
        elif test_type == 'both':
            mask_t = df['paired_t_pvalue'] < significance_level if 'paired_t_pvalue' in df.columns else False
            mask_w = df['wilcoxon_pvalue'] < significance_level if 'wilcoxon_pvalue' in df.columns else False
            mask = mask_t | mask_w
        else:
            print(f"Warning: test_type '{test_type}' not recognized or columns not found")
            continue
        
        significant_df = df[mask].copy()
        if len(significant_df) > 0:
            significant_dfs[comparison] = significant_df
        
        print(f"{comparison}: {len(significant_df)}/{len(df)} significant results")
    
    return significant_dfs

# Example usage function
def process_statistical_results(filepath: str, output_dir: str = "comparison_results"):
    """
    Complete workflow to process statistical results file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing statistical results
    output_dir : str
        Directory to save output files
    """
    
    # Load data
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Separate by comparison
    print("\n" + "="*50)
    print("SEPARATING BY MODALITY COMPARISON")
    print("="*50)
    comparison_dfs = separate_by_modality_comparison(
        df, 
        output_dir=output_dir,
        save_individual_files=True
    )
    
    # Create summary
    print("\n" + "="*50)
    print("CREATING SUMMARY TABLE")
    print("="*50)
    summary_df = create_summary_table(comparison_dfs)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Filter significant results
    print("\n" + "="*50)
    print("FILTERING SIGNIFICANT RESULTS")
    print("="*50)
    significant_dfs = filter_significant_results(comparison_dfs)
    
    # Save significant results
    for comparison, sig_df in significant_dfs.items():
        safe_filename = comparison.replace(' ', '_').replace('vs', 'vs').replace('/', '_')
        sig_path = os.path.join(output_dir, f"{safe_filename}_significant_only.csv")
        sig_df.to_csv(sig_path, index=False)
        print(f"Significant results for {comparison} saved to: {sig_path}")
    
    return comparison_dfs, summary_df, significant_dfs

def save_comparison_dfs_to_csv(
    comparison_dfs: Dict[str, pd.DataFrame], 
    output_method: str = 'individual',
    output_dir: str = 'comparison_outputs',
    combined_filename: str = 'all_comparisons.csv'
) -> None:
    """
    Save comparison DataFrames to CSV file(s).
    
    Parameters:
    -----------
    comparison_dfs : Dict[str, pd.DataFrame]
        Dictionary of comparison DataFrames from separate_by_modality_comparison()
    output_method : str
        'individual' - Save each comparison as separate CSV
        'combined' - Save all comparisons in one CSV with comparison column
        'both' - Save both individual and combined files
    output_dir : str
        Directory to save the files
    combined_filename : str
        Filename for combined CSV (only used if output_method includes 'combined')
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if output_method in ['individual', 'both']:
        print("Saving individual CSV files...")
        for comparison, df in comparison_dfs.items():
            # Clean filename
            safe_filename = comparison.replace(' ', '_').replace('vs', 'vs').replace('/', '_').replace('\\', '_')
            filepath = os.path.join(output_dir, f"{safe_filename}.csv")
            df.to_csv(filepath, index=False)
            print(f"  Saved: {filepath} ({len(df)} rows)")
    
    if output_method in ['combined', 'both']:
        print("Saving combined CSV file...")
        # Combine all DataFrames
        combined_df = pd.concat(comparison_dfs.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, combined_filename)
        combined_df.to_csv(combined_path, index=False)
        print(f"  Saved: {combined_path} ({len(combined_df)} rows total)")
    
    print(f"All files saved to directory: {output_dir}")

def export_comparison_tables_formatted(
    comparison_dfs: Dict[str, pd.DataFrame],
    output_dir: str = 'formatted_tables',
    include_significance_only: bool = True
) -> None:
    """
    Export comparison tables in publication-ready format.
    
    Parameters:
    -----------
    comparison_dfs : Dict[str, pd.DataFrame]
        Dictionary of comparison DataFrames
    output_dir : str
        Directory to save formatted tables
    include_significance_only : bool
        Whether to also create significance-only versions
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for comparison, df in comparison_dfs.items():
        # Create formatted version
        formatted_df = df.copy()
        
        # Format p-values
        if 'paired_t_pvalue' in formatted_df.columns:
            formatted_df['t_test_p_formatted'] = formatted_df['paired_t_pvalue'].apply(
                lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"
            )
            # Add significance stars
            if 't_test_stars' in formatted_df.columns:
                formatted_df['t_test_p_with_stars'] = (
                    formatted_df['t_test_p_formatted'] + formatted_df['t_test_stars']
                )
        
        if 'wilcoxon_pvalue' in formatted_df.columns:
            formatted_df['wilcoxon_p_formatted'] = formatted_df['wilcoxon_pvalue'].apply(
                lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"
            )
            # Add significance stars
            if 'wilcoxon_stars' in formatted_df.columns:
                formatted_df['wilcoxon_p_with_stars'] = (
                    formatted_df['wilcoxon_p_formatted'] + formatted_df['wilcoxon_stars']
                )
        
        # Round statistical values
        if 'paired_t_stat' in formatted_df.columns:
            formatted_df['paired_t_stat'] = formatted_df['paired_t_stat'].round(3)
        
        # Clean variable names
        if 'variable' in formatted_df.columns:
            formatted_df['variable_clean'] = formatted_df['variable'].str.replace('_', ' ').str.title()
        
        # Save formatted table
        safe_filename = comparison.replace(' ', '_').replace('vs', 'vs').replace('/', '_')
        formatted_path = os.path.join(output_dir, f"{safe_filename}_formatted.csv")
        formatted_df.to_csv(formatted_path, index=False)
        print(f"Formatted table saved: {formatted_path}")
        
        # Save significance-only version if requested
        if include_significance_only:
            if 't_test_significant' in formatted_df.columns or 'wilcoxon_significant' in formatted_df.columns:
                # Filter for significant results
                mask_t = formatted_df.get('t_test_significant', False)
                mask_w = formatted_df.get('wilcoxon_significant', False)
                significant_mask = mask_t | mask_w
                
                if significant_mask.any():
                    sig_df = formatted_df[significant_mask]
                    sig_path = os.path.join(output_dir, f"{safe_filename}_significant_formatted.csv")
                    sig_df.to_csv(sig_path, index=False)
                    print(f"Significant-only table saved: {sig_path} ({len(sig_df)} rows)")

# Quick usage functions
def quick_save_individual_csvs(comparison_dfs: Dict[str, pd.DataFrame], output_dir: str = 'individual_comparisons'):
    """Quick function to save each comparison as individual CSV."""
    save_comparison_dfs_to_csv(comparison_dfs, output_method='individual', output_dir=output_dir)

def quick_save_combined_csv(comparison_dfs: Dict[str, pd.DataFrame], filename: str = 'all_comparisons.csv'):
    """Quick function to save all comparisons in one CSV."""
    save_comparison_dfs_to_csv(comparison_dfs, output_method='combined', combined_filename=filename)

# If running as script
if __name__ == "__main__":
    # Example usage - uncomment and modify path as needed
    filepath = "filtered_statistical_testing_results_NEW_VIVA.csv"
    comparison_dfs, summary_df, significant_dfs = process_statistical_results(filepath)
    # Option 2: Save all in one combined file
    save_comparison_dfs_to_csv(comparison_dfs, output_method='combined', combined_filename='all_results_NEW_VIVA.csv')

    # Option 3: Get publication-ready formatted tables
    export_comparison_tables_formatted(comparison_dfs, output_dir='publication_tables_NEW_VIVA')
    # print("Functions loaded successfully!")
    # print("\nTo save comparison_dfs as CSV:")
    # print("1. Individual files: save_comparison_dfs_to_csv(comparison_dfs, 'individual')")
    # print("2. Combined file: save_comparison_dfs_to_csv(comparison_dfs, 'combined')")
    # print("3. Both: save_comparison_dfs_to_csv(comparison_dfs, 'both')")
    # print("4. Quick individual: quick_save_individual_csvs(comparison_dfs)")
    # print("5. Quick combined: quick_save_combined_csv(comparison_dfs)")
    # print("6. Formatted tables: export_comparison_tables_formatted(comparison_dfs)")