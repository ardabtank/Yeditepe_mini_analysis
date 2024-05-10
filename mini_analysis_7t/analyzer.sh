#!/bin/bash

FILES_DIR=$1

if [ ! -d "$FILES_DIR" ]; then
    echo "Directory $FILES_DIR not found!"
    exit 1
fi

DIRECTORY_NAME="analyses"

if [ ! -d "${FILES_DIR}/Data/$DIRECTORY_NAME" ]; then
    # Create the directory
    mkdir "${FILES_DIR}/Data/$DIRECTORY_NAME"
    echo "Directory 'Data/$DIRECTORY_NAME' created successfully."
fi

if [ ! -d "${FILES_DIR}/MC/$DIRECTORY_NAME" ]; then
    # Create the directory
    mkdir "${FILES_DIR}/MC/$DIRECTORY_NAME"
    echo "Directory 'MC/$DIRECTORY_NAME' created successfully."
fi
cd "${FILES_DIR}/Data" || exit
for file in *; do
    if [ -f "$file" ]; then   
        echo "Processing file: $file"
        runnner.exe "$file" "${FILES_DIR}/analyses/out$file"       
    fi
done

cd "${FILES_DIR}/analyses" || exit
hadd outdata_all.root *

echo "All analyzed files have been added to outdata_all.root"
cd ".." || exit
cd ".." || exit
cd "${FILES_DIR}/MC" || exit
for file in *; do
    if [ -f "$file" ]; then   
        runnner.exe "$file" "${FILES_DIR}/analyses/out$file"   
    fi
done
