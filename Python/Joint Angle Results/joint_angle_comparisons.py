import os
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def collect_and_process_file_paths(root_folder, column_name_hashmap):
    processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'minimum': [],
        'maximum': [],
        'range': []
    }))))
    
    print(f"Scanning root folder: {root_folder}")
    
    for modality in os.listdir(root_folder):
        modality_path = os.path.join(root_folder, modality)
        if not os.path.isdir(modality_path):
            continue
        
        print(f"Processing modality: {modality}")
        
        for participant in os.listdir(modality_path):
            participant_path = os.path.join(modality_path, participant)
            if not os.path.isdir(participant_path):
                continue
            
            # print(f"Processing participant: {participant}")
            
            category = "MoCap" if "MoCap" in participant else "NC"
            
            for dirpath, dirnames, filenames in os.walk(participant_path):
                for filename in filenames:
                    if "static" in filename.lower():
                        print(f"Skipping static file: {filename}")
                        continue
                    
                    key = os.path.splitext(filename)[0]
                    full_path = os.path.join(dirpath, filename)
                    file_ext = os.path.splitext(full_path)[1]

                    # Process specific files
                    if key in ["Squat", "Sqt To Box", "Sit To Stand"]:
                        # print(f"Processing {key} file for {participant}")
                        try:
                            if file_ext == '.csv':
                                df = pd.read_csv(full_path)
                            elif file_ext == '.xlsx':
                                df = pd.read_excel(full_path)
                            else:
                                raise ValueError(f"Unsupported file extension: {file_ext}")

                            # Rename columns based on the hashmap
                            df = df.rename(columns=column_name_hashmap)

                            for column in df.columns:
                                if column in column_name_hashmap.values():
                                    min_val = df[column].min()
                                    max_val = df[column].max()
                                    range_val = max_val - min_val

                                    processed_data[modality][category][key][column]['minimum'].append(min_val)
                                    processed_data[modality][category][key][column]['maximum'].append(max_val)
                                    processed_data[modality][category][key][column]['range'].append(range_val)

                                    # print(f"Processed column: {column}")

                        except Exception as e:
                            print(f"Error processing file {filename}: {str(e)}")
    
    return processed_data

def flatten_processed_data(processed_data):
    data = []
    
    for modality, categories in processed_data.items():
        for category, actions in categories.items():
            for action, columns in actions.items():
                for column, stats in columns.items():
                    for stat, values in stats.items():
                        for value in values:
                            data.append({
                                'Modality': modality,
                                'Category': category,
                                'Action': action,
                                'Column': column,
                                'Stat': stat,
                                'Value': value
                            })
    print(pd.DataFrame(data))
    return pd.DataFrame(data)

import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots(df):
    # Print unique values in 'Modality' to debug
    print("Unique modalities before processing:", df['Modality'].unique())
    
    actions = df['Action'].unique()
    categories = df['Category'].unique()  # Get unique categories
    
    # Filter for only Left and Right Knee angles
    knee_angles = ['Left Knee', 'Right Knee']
    df_filtered = df[df['Column'].isin(knee_angles)]
    
    # Print filtered dataframe head to debug
    print("Filtered dataframe head:\n", df_filtered.head())
    
    for category in categories:
        category_df = df_filtered[df_filtered['Category'] == category]
        
        for action in actions:
            action_df = category_df[category_df['Action'] == action]
            
            for stat in ['minimum', 'maximum', 'range']:
                stat_df = action_df[action_df['Stat'] == stat].copy()  # Create a copy to avoid SettingWithCopyWarning
                
                # Ensure 'Modality' column is of string type using .loc
                stat_df.loc[:, 'Modality'] = stat_df['Modality'].astype(str)
                
                # Print unique modalities in filtered dataframe for each stat
                print(f"Unique modalities for {category} - {action} - {stat}:", stat_df['Modality'].unique())
                
                # Create a larger figure with controlled size
                fig, ax = plt.subplots(figsize=(16, 10))  # Adjusted size for fewer columns
                
                # Create the boxplot
                sns.boxplot(
                    data=stat_df, 
                    x='Column', 
                    y='Value', 
                    hue='Modality', 
                    ax=ax,
                    dodge=True,
                    palette="Set2"
                )
                
                # Customize the plot
                ax.set_title(f'{stat.capitalize()} Knee Angle Boxplots for {category} - Action: {action}', fontsize=20, pad=20)
                ax.set_ylabel('Angle (degrees)', fontsize=16, labelpad=10)
                ax.set_xlabel('Knee', fontsize=16, labelpad=10)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Move the legend outside the plot and adjust its font size
                legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12, title_fontsize=14)
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Adjust layout to fit the legend
                plt.tight_layout()
                
                # Adjust the subplot parameters to give the legend more space
                plt.subplots_adjust(right=0.85)
                
                # Save the figure with enough space for the legend
                plt.savefig(f'{category}_{action}_{stat}_knee_boxplot.png', bbox_inches='tight', dpi=300)
                
                plt.close(fig)  # Close the figure to free up memory



# def plot_boxplots(df):
#     # Print unique values in 'Modality' to debug
#     print("Unique modalities before processing:", df['Modality'].unique())
    
#     actions = df['Action'].unique()
    
#     # Filter for only Left and Right Knee angles
#     knee_angles = ['Left Knee', 'Right Knee']
#     df_filtered = df[df['Column'].isin(knee_angles)]
    
#     # Print filtered dataframe head to debug
#     print("Filtered dataframe head:\n", df_filtered.head())
    
#     for action in actions:
#         action_df = df_filtered[df_filtered['Action'] == action]
        
#         for stat in ['minimum', 'maximum', 'range']:
#             stat_df = action_df[action_df['Stat'] == stat].copy()  # Create a copy to avoid SettingWithCopyWarning
            
#             # Ensure 'Modality' column is of string type using .loc
#             stat_df.loc[:, 'Modality'] = stat_df['Modality'].astype(str)
            
#             # Print unique modalities in filtered dataframe for each stat
#             print(f"Unique modalities for {action} - {stat}:", stat_df['Modality'].unique())
            
#             # Create a larger figure with controlled size
#             fig, ax = plt.subplots(figsize=(16, 10))  # Adjusted size for fewer columns
            
#             # Create the boxplot
#             sns.boxplot(
#                 data=stat_df, 
#                 x='Column', 
#                 y='Value', 
#                 hue='Modality', 
#                 ax=ax,
#                 dodge=True,
#                 palette="Set2"
#             )
            
#             # Customize the plot
#             ax.set_title(f'{stat.capitalize()} Knee Angle Boxplots for Action: {action}', fontsize=20, pad=20)
#             ax.set_ylabel('Angle (degrees)', fontsize=16, labelpad=10)
#             ax.set_xlabel('Knee', fontsize=16, labelpad=10)
#             ax.tick_params(axis='both', which='major', labelsize=12)
            
#             # Move the legend outside the plot and adjust its font size
#             legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12, title_fontsize=14)
            
#             # Remove top and right spines
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
            
#             # Adjust layout to fit the legend
#             plt.tight_layout()
            
#             # Adjust the subplot parameters to give the legend more space
#             plt.subplots_adjust(right=0.85)
            
#             # Save the figure with enough space for the legend
#             plt.savefig(f'{action}_{stat}_knee_boxplot.png', bbox_inches='tight', dpi=300)
            
#             plt.close(fig)  # Close the figure to free up memory




# Usage
root_folder = "/media/kaiarmstrong/HDD2T/SPORTS_DATA/Joint Angle Results"
column_name_hashmap = {
    "left_knee_flexion": "Left Knee",
    "LeftKneeAngle": "Left Knee",
    "right_knee_flexion": "Right Knee",
    "RightKneeAngle": "Right Knee",
    "left_ankle_flexion": "Left Ankle",
    "LeftAnkleAngle": "Left Ankle",
    "right_ankle_flexion": "Right Ankle",
    "RightAnkleAngle": "Right Ankle"
}

processed_data = collect_and_process_file_paths(root_folder, column_name_hashmap)
df = flatten_processed_data(processed_data)
plot_boxplots(df)

