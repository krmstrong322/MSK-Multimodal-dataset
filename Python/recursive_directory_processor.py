import os
from biomechanics_rename import standardize_column_names

def process_directory_tree(input_directory, output_directory=None):
    """
    Recursively process all files in a directory tree, standardizing column names
    in Excel or CSV files.

    Args:
        input_directory (str): Path to the root directory to process.
        output_directory (str, optional): Path to save the output files. If None, 
                                          saves in the same directory as the input files.
    """
    for root, _, files in os.walk(input_directory):
        for file_name in files:
            if file_name.lower().endswith(('.xlsx', '.xls', '.csv')):
                input_file = os.path.join(root, file_name)
                
                # Determine output directory and file path
                if output_directory:
                    relative_path = os.path.relpath(root, input_directory)
                    output_subdir = os.path.join(output_directory, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, f"{os.path.splitext(file_name)[0]}_standardized{os.path.splitext(file_name)[1]}")
                else:
                    output_file = os.path.join(root, f"{os.path.splitext(file_name)[0]}_standardized{os.path.splitext(file_name)[1]}")
                
                # Process the file
                try:
                    result = standardize_column_names(input_file, output_file)
                    print(f"Standardized file saved to: {result}")
                except Exception as e:
                    print(f"Failed to process {input_file}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recursively standardize column names in Excel or CSV files.")
    parser.add_argument("input_directory", type=str, help="Path to the root input directory.")
    parser.add_argument("--output_directory", type=str, help="Path to save the output files. Defaults to input directory.", default=None)

    args = parser.parse_args()

    process_directory_tree(args.input_directory, args.output_directory)

    # python recursive_directory_processor.py "C:\Users\Kai Armstrong\Downloads\SPORTS_DATA\Joint Angle Results"