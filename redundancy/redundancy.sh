#!/bin/bash

input=$1
if [ -z "$input" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi
if [ ! -f "$input" ]; then
    echo "Input file not found: $input"
    exit 1
fi


xz --decompress --stdout "$input"  \
    | python3 redundancy.py --hash --file "$input"

# xz --decompress --stdout "$input"  \
#     | python3 rdautils/compression/distill.py --from compress --to canonical \
#     | python3 redundancy.py --hash --file "$input"