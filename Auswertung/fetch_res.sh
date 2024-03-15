#!/bin/bash

# Prompt for folder name
read -p "Enter folder name: " folder_name
read -p "Enter model name: " model_name

# Check if folder name is provided
if [ -z "$folder_name" ]; then
    echo "Error: Please provide a folder name."
    exit 1
fi

# Remote server details
remote_user="tu_epajg01"
remote_server="login01.binac.uni-tuebingen.de"
remote_base_path="/beegfs/work/tu_epajg01/Python/Corellation/$folder_name"
remote_directories=("dependencies" "Virtual_Reality/model_data" "output" "Ensemble/template_model")

# Destination directory
destination_dir="$PWD/$folder_name/$model_name"

# Create local destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Construct the scp command to copy all directories at once
scp_command="scp -r $remote_user@$remote_server:'"
for ((i=0; i<${#remote_directories[@]}; i++)); do
    dir="${remote_directories[i]}"
    scp_command+="$remote_base_path/$model_name/$dir"
    if [ $i -lt $((${#remote_directories[@]} - 1)) ]; then
        scp_command+=" "
    fi
done

scp_command+="' $destination_dir"

# Print the scp command
#echo "SCP Command: $scp_command"

# Print the length of the scp command
#echo "Length of SCP Command: ${#scp_command}"

# Run scp command
echo "Copying directories..."
eval $scp_command

# Check if scp command was successful
if [ $? -eq 0 ]; then
    echo "Directories copied successfully."
else
    echo "Failed to copy directories."
fi

