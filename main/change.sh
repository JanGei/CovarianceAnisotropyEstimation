#!/bin/bash

# Loop through each subdirectory in the current directory
for dir in */; do
    # Get the name of the subdirectory
    subdirectory_name=$(basename "$dir")
    
    # Define the file to be modified
    target_file="${dir}main_binnac.sh"

    # Check if the file exists
    if [ -f "$target_file" ]; then
        # Find the current path that needs replacement by matching the known structure
        current_path=$(grep -oP "/beegfs/work/tu_epajg01/Python/Corellation/LocalBatch2/\K[^/]+" "$target_file")

        # Use sed to replace the dynamically found string with the subdirectory name
        sed -i "s|/beegfs/work/tu_epajg01/Python/Corellation/LocalBatch2/${current_path}/main.py|/beegfs/work/tu_epajg01/Python/Corellation/LocalBatch2/${subdirectory_name}/main.py|" "$target_file"
        
        echo "Updated ${target_file} with subdirectory name ${subdirectory_name}"
    else
        echo "File ${target_file} not found in ${dir}, skipping."
    fi
done
