#!/bin/bash

process_directory() 
{
    # -----------------------------------------------------------------------------------------------------
    # Preprocessing

    directory="$1"
    dirname=$(basename "$directory")
    
    python3 ./segmentation/algoVSts.py "$dirname" --use_minmax
    python3 ./segmentation/algoVSts.py "$dirname" --use_mean
    python3 ./segmentation/hfuVSts.py "$dirname" --use_minmax
    python3 ./segmentation/hfuVSts.py "$dirname" --use_mean

    # -----------------------------------------------------------------------------------------------------
    # Ending this case and moving on to another
    echo "Processed $dirname" >> ./segpros.txt

    echo ' =============================='
}

source /tbcrohn_py39/bin/activate
rm -rvf ./segpros.txt
rm -rvf ./seg.json


# Process directories in each category
superdirectory="/home/medicalai/cd_itb/dataset"

echo 'Working on iTB'
for dirpath in "$superdirectory/itb"/*/; do
    if [ -d "$dirpath" ]; then
        process_directory "$dirpath"
        echo "Processed $dirpath"
    fi
done

echo 'Working on Crohns:'
for dirpath in "$superdirectory/crohn"/*/; do
    if [ -d "$dirpath" ]; then
        process_directory "$dirpath"
        echo "Processed $dirpath"

    fi
done

echo 'Working on Normal'
for dirpath in "$superdirectory/normal"/*/; do
    if [ -d "$dirpath" ]; then
        process_directory "$dirpath"
        echo "Processed $dirpath"
    fi
done

# Ending
touch "Done_WITH_SEG"