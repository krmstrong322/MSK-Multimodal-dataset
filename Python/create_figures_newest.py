import pandas as pd


#create figures


# Load the filtered results

file = "filtered_kinematics_summary_NEW_renamed.csv"
data = pd.read_csv(file)

# drop rows with "Balance" and "Lunge" in the 'action' column
data = data[~data['action'].str.contains('Balance|Lunge|Static', case=False, na=False)]

# drop rows with "MediapipeFused" and "VIBEFused" in the 'measurement_system' column
data = data[~data['measurement_system'].str.contains('MediapipeFused|VIBEFused', case=False, na=False)]

# drop columns with "ankle" in the column name
data = data.loc[:, ~data.columns.str.contains('ankle', case=False, na=False)]

# drop column called 'file_path'
data = data.drop(columns=['file_path'], errors='ignore')

# print(data.head())

# Save as new csv file
data.to_csv("filtered_kinematics_summary_NEW_renamed.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('filtered_kinematics_summary_NEW_renamed.csv')

# Filter for MoCap condition only
df_mocap = df[df['condition'] == 'Tight']

# Define the metrics to plot
metrics = ['left_knee_angle_min', 'left_knee_angle_max', 'left_knee_angle_rom',
           'right_knee_angle_min', 'right_knee_angle_max', 'right_knee_angle_rom']

# Define the actions
actions = df_mocap['action'].unique()

# Set up the plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create a box plot for each action/metric combination
for action in actions:
    if pd.isna(action):  # Skip NaN actions
        continue
        
    action_data = df_mocap[df_mocap['action'] == action]
    
    for metric in metrics:
        # Filter out rows with missing data for this metric
        plot_data = action_data.dropna(subset=[metric])
        
        if len(plot_data) == 0:
            continue
            
        # Create the plot
        plt.figure()
        ax = sns.boxplot(x='measurement_system', y=metric, 
                         data=plot_data, palette="Set3")
        
        # Add individual data points
        sns.stripplot(x='measurement_system', y=metric, 
                      data=plot_data, dodge=True, edgecolor='gray', 
                      linewidth=1, palette="Set3", ax=ax)
        
        # Improve the legend
        handles, labels = ax.get_legend_handles_labels()
        l = plt.legend(handles, labels, title='Measurement System')
        
        # Customize the plot
        plt.title(f'{action} - {metric.replace("_", " ").title()}')
        plt.xlabel('Measurement System')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=0)
        
        # Adjust layout to prevent title/axis overlap
        plt.tight_layout()
        
        # Save the plot
        filename = f"{action}_{metric}_boxplot.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Saved plot: {filename}")

# Define the metric types (min, max, rom)
metric_types = ['angle_min', 'angle_max', 'angle_rom']
sides = ['left', 'right']

for action in actions:
    if pd.isna(action):
        continue

    action_data = df_mocap[df_mocap['action'] == action]

    for metric_type in metric_types:
        # Prepare data in long format for seaborn
        plot_data = []
        for side in sides:
            metric_col = f"{side}_knee_{metric_type}"
            if metric_col not in action_data.columns:
                continue
            temp = action_data[['measurement_system', metric_col]].copy()
            temp = temp.rename(columns={metric_col: 'value'})
            temp['Side'] = side.capitalize()
            plot_data.append(temp)
        if not plot_data:
            continue
        plot_data = pd.concat(plot_data, ignore_index=True).dropna(subset=['value'])

        if len(plot_data) == 0:
            continue

        plt.figure()
        # Each modality gets a different color, left/right are grouped on x-axis
        ax = sns.boxplot(
            x='Side', y='value', hue='measurement_system',
            data=plot_data, palette="Set2"
        )
        sns.stripplot(
            x='Side', y='value', hue='measurement_system',
            data=plot_data, dodge=True, edgecolor='gray',
            linewidth=1, palette="Set2", ax=ax, alpha=0.5, legend=False
        )

        # Remove duplicate legends from stripplot
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Measurement System')

        plt.title(f'{action} - Knee {metric_type.split("_")[1].title()}')
        plt.xlabel('Side')
        plt.ylabel(f'Knee {metric_type.split("_")[1].title()}')
        plt.xticks(rotation=0)
        plt.tight_layout()

        filename = f"{action}_knee_{metric_type}_boxplot.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved plot: {filename}")

print("All plots generated successfully!")