import os
import cv2
import mediapipe as mp
import pandas as pd


body_parts = [
    "nose",
    "inner_eye_l",
    "eye_l",
    "outer_eye_l",
    "inner_eye_r",
    "eye_r",
    "outer_eye_r",
    "ear_l",
    "ear_r",
    "mouth_l",
    "mouth_r",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "pinky_l",
    "pinky_r",
    "index_finger_l",
    "index_finger_r",
    "thumb_l",
    "thumb_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "heel_l",
    "heel_r",
    "foot_index_l",
    "foot_index_r",
]

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Initialize Mediapipe Pose
    pose = mp.solutions.pose
    pose_instance = pose.Pose()

    full_keypoints = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Process the image with Mediapipe Pose
        results = pose_instance.process(image)

        # Extract keypoints
        keypoints = []
        for data_point in results.pose_landmarks.landmark:
            keypoints.append({
                'X': data_point.x,
                'Y': data_point.y,
                'Z': data_point.z,
            })
        full_keypoints.append(keypoints)

    cap.release()
    #print(full_keypoints)
    #print(results)
    #print(results.pose_landmarks)

    df = pd.DataFrame(full_keypoints, columns=body_parts)

    reshaped_data = {}

    # Iterate through the original DataFrame and reshape the data
    for col_name, col_data in df.items():
        body_part = col_name.split(' - ')[-1]  # Extract the body part name
        for row_idx, cell in enumerate(col_data):
            if row_idx not in reshaped_data:
                reshaped_data[row_idx] = {}
            reshaped_data[row_idx][f'{body_part}_x'] = cell['X']
            reshaped_data[row_idx][f'{body_part}_y'] = cell['Y']
            reshaped_data[row_idx][f'{body_part}_z'] = cell['Z']

    # Create a new DataFrame from the reshaped data
    reshaped_df = pd.DataFrame.from_dict(reshaped_data, orient='index')
    #reshaped_df.to_csv(f'{output_folder}{vid_input}_keypoints.csv', index=False)
    return reshaped_df

def process_videos_in_directory(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter out non-video files (you can customize the video extensions)
    video_files = [file for file in files if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]

    # Loop through each video file
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        keypoints = process_video(video_path)
        output_file = video_path.replace('.avi', '_mediapipe.csv')
        keypoints.to_csv(output_file, index=False)
        # Print or process the keypoints as needed

if __name__ == "__main__":
    # List of directories
    directories = [
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p01",
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p02",
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p03",
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p04",
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p05",
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p06",
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p07",
        "/media/kaiarmstrong/HDD2T/SPORTS_DATA/p08"
    ]

    for directory_path in directories:
        # Check if the directory exists
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found - {directory_path}")
        else:
            # Process videos in the specified directory
            process_videos_in_directory(directory_path)