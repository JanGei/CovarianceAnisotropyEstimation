#!/bin/bash

old_name=45
new_name=45

# Define the line you want to replace and the new line content
old_line="setup = computer\[0\]"
new_line="setup = computer\[1\]"

# Iterate over each directory in the current directory
#for dir in */; do
#  # Remove trailing slash
#  dir=${dir%/}
#  
#  # Check if the directory name contains "old_name"
#  if [[ "$dir" == *"$old_name"* ]]; then
#    # Create the new directory name by replacing "old_name" with "new_name"
#    new_dir="${dir//$old_name/$new_name}"
#    
#    # Copy the directory to the new directory with the renamed name
#    cp -r "$dir" "$new_dir"
#    
#    echo "Copied $dir to $new_dir"
#  fi
#done

# Define the letter combination to filter directories
pattern="*$new_name*/"

# Loop through each directory in the current working directory that matches the pattern
for dir in $pattern; do
    # Remove trailing slash
    dir=${dir%/}

    # Check if the dependencies/model_params.py file exists
    target_file="${dir}/dependencies/model_params.py"
    if [ -f "$target_file" ]; then
        echo "Processing $target_file..."
        
        # Check if the line exists in the file
        if grep -q "$old_line" "$target_file"; then
            # Use sed to replace the line while preserving indentation
            sed -i -e "s/^\( *\)${old_line}.*/\1${new_line}/" "$target_file"
            echo "Updated ${target_file}"
        else
            echo "No matching line found in ${target_file}, skipping replacement."
        fi
    else
        echo "File ${target_file} not found in ${dir}, skipping."
    fi
done

