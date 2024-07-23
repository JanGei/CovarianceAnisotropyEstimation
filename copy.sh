#!/bin/bash

# Iterate over each directory in the current directory
for dir in */; do
  # Check if the directory name contains "25"
  if [[ "$dir" == *25* ]]; then
    # Create the new directory name by replacing "25" with "50"
    new_dir="${dir//25/50}"
    
    # Remove the trailing slash from the directory names
    dir=${dir%/}
    new_dir=${new_dir%/}
    
    # Copy the directory to the new directory with the renamed name
    cp -r "$dir" "$new_dir"
    
    echo "Copied $dir to $new_dir"
  fi
done

