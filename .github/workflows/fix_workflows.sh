#!/bin/bash

# This script fixes all gcolab workflow files to properly download executed notebooks from Colab

for file in gcolab__*.yml; do
    echo "Fixing: $file"
    
    # Replace the download and copy logic with proper notebook download
    sed -i \
        's|safe=$(echo "$nb" | sed '\''s#\[/ ()+\]#_#g'\'' | sed '\''s#\.ipynb$##'')|safe=$(echo "$nb" | sed '\''s#[/ ()+]#_#g'\'' \| sed '\''s#\\.ipynb$##'\'')|g' \
        "$file"
    
    # Replace the old copy command with proper download
    sed -i \
        's|cp "$nb" "artifacts/\${safe}_executed.ipynb" \|\| true|nb_basename=$(basename "$nb")\n            python colab_patched.py --auth adc download -s github_job "/content/$nb_basename" "artifacts/${safe}_executed.ipynb" 2>/dev/null \|\| cp "$nb" "artifacts/${safe}_executed.ipynb"|g' \
        "$file"
done

echo "All workflow files updated!"
