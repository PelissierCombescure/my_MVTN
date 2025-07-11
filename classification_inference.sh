#!/bin/bash

# Loop from 4 to 11 (inclusive)
for i in $(seq 0 11); do
  echo "\nRunning classification_inference.py with parameter -one_by_one $i\n"
  python3 classification_inference.py -one_by_one "$i"

  echo "\nFinished running classification_inference.py with parameter -one_by_one $i\n"
  echo "----------------------------------------"
done