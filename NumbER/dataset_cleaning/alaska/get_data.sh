#!/bin/bash

# Set the directory and file to search
dir="./notebook/notebook_specs"
output_file="./X3_data.json"

# Write the opening bracket of the JSON array to the output file
echo "[" > "$output_file"

# Set a counter for the progress
counter=0

# Iterate over each row in the input file
while read -r row; do
  # Search for the row value in all files within the directory
  result=$(grep -rnw "$dir" -e "$row" | head -n 1)
  
  # If a result is found, extract the JSON and write it to the output file
  if [ -n "$result" ]; then
    file_content=$(echo "$result" | awk -F ":" '{print $1}' | xargs cat)
    echo "$file_content," >> "$output_file"
  else
    # If no files are found, print the row
    echo "No files found for row: $row"
  fi
  
  # Print the progress
  counter=$((counter+1))
  echo "Progress: $counter rows searched"
done < "$1"

# Remove the trailing comma from the last JSON object in the output file
sed -i '$ s/,$//' "$output_file"

# Write the closing bracket of the JSON array to the output file
echo "]" >> "$output_file"