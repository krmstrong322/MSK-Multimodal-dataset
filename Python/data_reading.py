import os
import pandas as pd
import numpy as np
import re
import glob
from pathlib import Path

# Define the lookup table as a dictionary
lookup_table = {
    # Substring: String
    "Static": "Static",
    "static": "Static",
    "Squat": "Squat",
    "squat": "Squat",
    "SquatToBox": "SquatToBox",
    "squattobox": "SquatToBox",
    "squat_to_box": "SquatToBox",
    "Squat_To_Box": "SquatToBox",
    "Sqt To Box": "SquatToBox",
    "SqtToBox": "SquatToBox",
    "SitToStand": "SitToStand",
    "SitToStnd": "SitToStand",
    "sit_to_stand": "SitToStand",
    "Sit_To_Stand": "SitToStand",
    "sittostand": "SitToStand",
    "Balance_Left": "LeftBalance",
    "left_balance": "LeftBalance",
    "leftbalance": "LeftBalance",
    "Left_Balance": "LeftBalance",
    "Right_Balance": "RightBalance",
    "Balance_Right": "RightBalance",
    "right_balance": "RightBalance",
    "rightbalance": "RightBalance",
    "rightlbalance": "RightBalance",
    "LeftLunges": "LeftLunges",
    "left_lunge": "LeftLunges",
    "Left_Lunge": "LeftLunges",
    "leftlunge": "LeftLunges",
    "Lunge_Left": "LeftLunges",
    "RightLunges": "RightLunges",
    "right_lunge": "RightLunges",
    "Right_Lunge": "RightLunges",
    "rightlunge": "RightLunges",
    "Lunge_Right": "RightLunges",
    "LeftForwardLunge": "LeftLunges",
    "RightForwardLunge": "RightLunges",
}

def determine_action(filename):
    """Determine the action based on the filename using the lookup table."""
    normalized_filename = filename.lower().replace("_", "").replace(" ", "")
    
    for key in lookup_table:
        normalized_key = key.lower().replace("_", "").replace(" ", "")
        if normalized_key in normalized_filename:
            return lookup_table[key]
    
    # Log unmatched filenames for debugging
    print(f"Unmatched filename: {filename}")
    return "Unknown"

def process_file(file_path):
    """Process a single file and extract min, max, and ROM for each kinematic variable."""
    try:
        # Check file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None
        
        # Check if the required columns exist
        required_cols = ["left_knee_angle", "right_knee_angle", "left_ankle_angle", "right_ankle_angle"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns in {file_path}: {missing_cols}")
            # Try to find similar column names
            available_cols = df.columns.tolist()
            for col in missing_cols:
                # Look for similar column names (case insensitive)
                potential_cols = [c for c in available_cols if col.lower().replace('_', '') in c.lower().replace('_', '')]
                if potential_cols:
                    print(f"  Potential matches for {col}: {potential_cols}")
                    # Replace with the first match
                    df[col] = df[potential_cols[0]]
            
            # Check again if we still have missing columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Still missing columns after attempted matching, skipping file: {file_path}")
                return None
        
        # Extract metrics for each column
        results = {}
        for col in required_cols:
            # Filter out NaN values
            values = df[col].dropna()
            if len(values) == 0:
                results[f"{col}_min"] = np.nan
                results[f"{col}_max"] = np.nan
                results[f"{col}_rom"] = np.nan
            else:
                results[f"{col}_min"] = values.min()
                results[f"{col}_max"] = values.max()
                results[f"{col}_rom"] = values.max() - values.min()
        
        return results
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def extract_participant_info(file_path):
    """Extract participant ID and condition from file path."""
    # Extract participant ID (P01, P02, etc.)
    participant_match = re.search(r'P0*(\d+)', os.path.dirname(file_path))
    participant_id = f"P{int(participant_match.group(1)):02d}" if participant_match else "Unknown"

    # Extract condition (Tight or Loose)
    condition = "Unknown"
    if "Tight" in os.path.dirname(file_path):
        condition = "Tight"
    elif "Loose" in os.path.dirname(file_path):
        condition = "Loose"

    return participant_id, condition

def extract_measurement_system(file_path):
    """Extract the measurement system from the file path."""
    path = Path(file_path)
    # Go up at least 2 levels to find the measurement system directory
    parent_dirs = path.parts
    for part in parent_dirs:
        if "IMU" in part:
            return "IMU"
        elif "Mediapipe" in part and "(1)" in part:
            return "Mediapipe1"
        elif "Mediapipe" in part and "(2)" in part:
            return "Mediapipe2"
        elif "Mediapipe" in part and "FUSED" in part:
            return "MediapipeFused"
        elif "MoCap" in part and not any(p in part for p in ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]):
            return "MoCap"
        elif "VIBE" in part and "(1)" in part:
            return "VIBE1"
        elif "VIBE" in part and "(2)" in part:
            return "VIBE2"
        elif "VIBE" in part and "FUSED" in part:
            return "VIBEFused"
    return "Unknown"

def main():
    # Base directory where all measurement system folders are located
    base_dir = "Joint Angle Results"  # Change this if needed
    
    # List of measurement system directories
    measurement_dirs = [
        "IMU Joint Angle Outputs",
        "Mediapipe Joint Angle Outputs (1)",
        "Mediapipe Joint Angle Outputs (2)",
        "MoCap Joint Angle Outputs",
        "VIBE Joint Angle Outputs (1)",
        "VIBE Joint Angle Outputs (2)"
    ]
    
    # Initialize a list to store all results
    all_results = []
    
    # Process each measurement system directory
    for meas_dir in measurement_dirs:
        meas_path = os.path.join(base_dir, meas_dir)
        if not os.path.exists(meas_path):
            print(f"Directory not found: {meas_path}")
            continue
        
        print(f"Processing {meas_dir}...")
        
        # Walk through all participant subdirectories
        for root, dirs, files in os.walk(meas_path):
            # Find all CSV and Excel files with "_standardized" in the filename
            for file_pattern in ["*_standardized*.csv", "*_standardized*.xlsx", "*_standardized*.xls"]:
                for file_path in glob.glob(os.path.join(root, file_pattern)):
                    print(f"Processing file: {file_path}")
                    
                    # Extract metadata
                    participant_id, condition = extract_participant_info(file_path)
                    measurement_system = extract_measurement_system(file_path)
                    action = determine_action(os.path.basename(file_path))
                    
                    # Process the file
                    results = process_file(file_path)
                    if results:
                        # Add metadata to results
                        results["participant_id"] = participant_id
                        results["condition"] = condition
                        results["measurement_system"] = measurement_system
                        results["action"] = action
                        results["file_path"] = file_path
                        
                        # Add to the list of all results
                        all_results.append(results)
    
    # Convert results to a DataFrame
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        cols = [
            "participant_id", "condition", "measurement_system", "action",
            "left_knee_angle_min", "left_knee_angle_max", "left_knee_angle_rom",
            "right_knee_angle_min", "right_knee_angle_max", "right_knee_angle_rom",
            "left_ankle_angle_min", "left_ankle_angle_max", "left_ankle_angle_rom",
            "right_ankle_angle_min", "right_ankle_angle_max", "right_ankle_angle_rom",
            "file_path"
        ]
        df_results = df_results[cols]
        
        # Save to CSV
        output_file = "kinematics_summary_NEW_renamed.csv"
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Basic statistics
        print("\nSummary statistics:")
        print(f"Total number of processed files: {len(df_results)}")
        print(f"Participants: {df_results['participant_id'].nunique()}")
        print(f"Measurement systems: {df_results['measurement_system'].nunique()}")
        print(f"Actions: {df_results['action'].nunique()}")
        
        # Generate a pivot table for easier analysis
        pivot_columns = [
            "participant_id", "condition", "measurement_system", "action"
        ]
        value_columns = [
            col for col in df_results.columns 
            if any(metric in col for metric in ["_min", "_max", "_rom"]) 
            and not col.endswith("file_path")
        ]
        
        # Save a more detailed Excel report with sheets for min, max, ROM
        with pd.ExcelWriter("kinematics_detailed_report.xlsx") as writer:
            # Save the raw data
            df_results.to_excel(writer, sheet_name="Raw_Data", index=False)
            
            # Create summary sheets for min, max, and ROM values
            for metric in ["min", "max", "rom"]:
                metric_cols = [col for col in value_columns if f"_{metric}" in col]
                if metric_cols:
                    # Create a pivot table for each metric
                    for joint in ["knee", "ankle"]:
                        joint_cols = [col for col in metric_cols if joint in col]
                        if joint_cols:
                            # Filter and pivot the data
                            summary_df = df_results[pivot_columns + joint_cols].copy()
                            
                            # Rename columns for better readability
                            summary_df.columns = [
                                col.replace(f"_{metric}", f" {metric.upper()}") 
                                if f"_{metric}" in col else col 
                                for col in summary_df.columns
                            ]
                            
                            # Save to sheet
                            sheet_name = f"{joint.capitalize()}_{metric.upper()}"
                            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
        print(f"Detailed report saved to kinematics_detailed_report.xlsx")
    else:
        print("No results were collected. Check the file paths and formats.")

if __name__ == "__main__":
    main()