#!/bin/bash

# Iterate over each directory in the current directory
for dir in */; do
  # Check if the directory name contains "p"
  if [[ "$dir" == *p* ]]; then
    # Enter the directory
    cd "$dir"
    
    # Execute qsub main_binnac.sh
    qsub main_binnac.sh
    
    # Return to the original directory
    cd ..
    
    echo "Executed qsub main_binnac.sh in $dir"
  fi
done

