#!/bin/bash

# Define the line you want to replace and the new line content
old_line="'Erdalx': False"
new_line="'Erdalx': True"

# Loop through each directory in the current working directory
for dir in */; do
    # Check if the dependencies/model_params.py file exists
    if [ -f "${dir}dependencies/model_params.py" ]; then
        # Use sed to replace the line in the file
        sed -i "s|${old_line}|${new_line}|" "${dir}dependencies/model_params.py"
        echo "Updated ${dir}dependencies/model_params.py"
    else
        echo "File ${dir}dependencies/model_params.py not found, skipping."
    fi
done

