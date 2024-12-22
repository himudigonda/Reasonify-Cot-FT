#!/bin/bash

process_file() {
    local file=$1
    if [[ $file == *"/__pycache__/"* ]] ||
        [[ $file == *"/xcuserdata/"* ]] ||
        [[ $file == *"/Assets.xcassets/"* ]] ||
        [[ $file == *"/DerivedData/"* ]] ||
        [[ $file == *"/temp_uploads/"* ]] ||
        [[ $file == *"/.git/"* ]] ||
        [[ $file == *"/Resources/"* ]] ||
        [[ $file == *"/*.pdf"* ]] ||
        [[ $file == *"/*.jpg"* ]] ||
        [[ $file == *"/*.png"* ]] ||
        [[ $file == *"/*.JPG"* ]] ||
        [[ $file == *"/*.jpeg"* ]] ||
        [[ $file == *"/build/"* ]] ||
        [[ $file == *".DS_Store"* ]] ||
        [[ $file == *".pytest_cache/"* ]]; then
        return
    fi
    if [ -d "$file" ]; then
        for f in "$file"/*; do
            process_file "$f"
        done
    elif [ -f "$file" ]; then

        echo "=== File: $file ==="
        echo "----------------------------------------"
        cat "$file"
        echo -e "\n----------------------------------------\n"
    fi
}

if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

process_file "$1"
