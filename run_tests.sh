#!/bin/bash

# Find all .py files in the config/tests directory
for file in config/tests/*.py
do
  # Call the python command with the file name as an argument
  python main.py --config "$file"
done