#!/bin/bash

# Output file name
output_file="combined_partisan_bias_tables.md"

# Array of input files in the specified order
files=(
    "partisan_bias_table_pop_minus.md"
    "partisan_bias_table_pop_plus.md"
    "partisan_bias_table_distpair.md"
    "partisan_bias_table_ust.md"
    "partisan_bias_table_distpair_ust.md"
    "partisan_bias_table_reversible.md"
    "partisan_bias_table_county25.md"
    "partisan_bias_table_county50.md"
    "partisan_bias_table_county75.md"
    "partisan_bias_table_county100.md"
)

# Clear the output file if it exists
> "$output_file"

# Process each file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        # Append the file contents
        cat "$file" >> "$output_file"
        # Add a newline and echo the filename
        echo "" >> "$output_file"
        echo "Source: $file" >> "$output_file"
        echo "" >> "$output_file"
    else
        echo "Warning: $file not found, skipping..."
    fi
done

echo "Files combined into $output_file"