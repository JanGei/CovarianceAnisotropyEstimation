#!/bin/bash

# Iterate over each directory in the current directory
for dir in */; do
   # Enter the directory
   cd "$dir"

   # Execute qsub main_binnac.sh
   qsub main_binnac.sh

   # Return to the original directory
   cd ..

   echo "Executed qsub main_binnac.sh in $dir"
done

