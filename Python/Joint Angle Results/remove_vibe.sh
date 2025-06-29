#!/bin/bash

# Function to remove files with "_vibe" at the end of the filename
remove_vibe_files() {
    local dir_path="$1"
    echo "Processing directory: $dir_path"
    find "$dir_path" -maxdepth 1 -type f -name '*_vibe' -print -delete
}

# Get the current directory
base_dir="$(pwd)"

echo "Starting in directory: $base_dir"

# Loop through all directories in the current directory
for dir in "$base_dir"/*/ ; do
    if [ -d "$dir" ]; then
        remove_vibe_files "$dir"
    fi
done

echo "Operation completed. Files with '_vibe' at the end of the filename have been removed from all subdirectories."
