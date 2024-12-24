import os
import pandas as pd

# Defining splits and field types
splits = ["train", "test", "valid"]
fields = ["diff", "msg", "repo", "sha", "time"]

# Function to convert txt files for a split into a CSV
def convert_to_csv(split):
    data = {field: [] for field in fields} 
    
    # Reading data from each field file
    for field in fields:
        filename = f"data/python/sort_random_train80_valid10_test10/{split}.{field}.txt"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data[field] = f.read().splitlines()  # Read all lines as a list
        else:
            print(f"Warning: {filename} not found.")
            return
    
    # Ensuring all files have the same number of lines
    num_samples = len(data[fields[0]])
    if not all(len(data[field]) == num_samples for field in fields):
        print(f"Error: Files for split '{split}' have inconsistent line counts.")
        return
    
    # Creating a DataFrame
    df = pd.DataFrame(data)
    
    output_filename = f"data/python/sort_random_train80_valid10_test10/{split}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")

# Converting each dataset split
for split in splits:
    convert_to_csv(split)
