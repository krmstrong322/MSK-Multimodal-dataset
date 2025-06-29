import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Define root directory
ROOT_DIR = Path("/your/root/directory")

# File extensions by modality
MODALITY_INFO = {
    "IMU Joint Angle Outputs": ".xlsx",
    "MoCap Joint Angle Outputs": ".xlsx",
    "Mediapipe Joint Angle Outputs (1)": ".csv",
    "Mediapipe Joint Angle Outputs (2)": ".csv",
    "VIBE Joint Angle Outputs (1)": ".csv",
    "VIBE Joint Angle Outputs (2)": ".csv"
}

def extract_range_of_motion(file_path, joint_columns):
    df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
    roms = {}
    for joint in joint_columns:
        if joint in df.columns:
            joint_data = df[joint].dropna()
            if not joint_data.empty:
                rom = joint_data.max() - joint_data.min()
                avg_max = joint_data.max()
                avg_min = joint_data.min()
                roms[joint] = (rom, avg_min, avg_max)
    return roms

def collect_data():
    records = []

    for modality, ext in MODALITY_INFO.items():
        modality_path = ROOT_DIR / modality
        for participant_folder in modality_path.glob("P*"):
            participant_id = participant_folder.name
            for action_file in participant_folder.glob(f"*{ext}"):
                action_name = action_file.stem
                rom_data = extract_range_of_motion(action_file, ['Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'])
                for joint, (rom, min_val, max_val) in rom_data.items():
                    records.append({
                        "Modality": modality,
                        "Participant": participant_id,
                        "Action": action_name,
                        "Joint": joint,
                        "RangeOfMotion": rom,
                        "Min": min_val,
                        "Max": max_val
                    })
    return pd.DataFrame(records)

def plot_boxplots(df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    for (modality, joint), group_df in df.groupby(["Modality", "Joint"]):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=group_df, x="Action", y="RangeOfMotion", hue="Participant")
        plt.title(f"{modality} - {joint}")
        plt.ylabel("Range of Motion (degrees)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"{save_dir}/{modality}_{joint}_RoM_Boxplot.png")
        plt.close()

# Run the full pipeline
df_all = collect_data()
plot_boxplots(df_all)
