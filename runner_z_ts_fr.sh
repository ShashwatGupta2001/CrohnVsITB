#!/bin/bash

process_directory() 
{
    # -----------------------------------------------------------------------------------------------------
    # Preprocessing

    directory="$1"
    disease="$2"
    
    start_time=$(date +%s)
    
    rm -rf ./dirs/segmentorviewer
    mkdir -p dirs/segmentorviewer
    dirname=$(basename "$directory")
    final_directory="./dirs/segmentorviewer/$dirname"
    mkdir -p "$final_directory"
    
    cp -r "$directory"/* "$final_directory"

    for file in "$final_directory"/*; do
        base_name=$(basename "$file")
        if [[ $base_name =~ ^I[0-9]+$ ]]; then
            mv "$file" "$file.dcm"
        else
            rm -rf "$file"
        fi
    done

    echo "$final_directory"
    TotalSegmentator -i "$final_directory" -o "./dirs/segmentations/$dirname" --roi_subset vertebrae_L4

    python3 ./generic/get_z_of_vertebrae.py "$dirname" 

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    python3 ./generic/find_white_z.py "$dirname" $elapsed_time $disease
    
    # -----------------------------------------------------------------------------------------------------
    # Total Segmentator

    start_time=$(date +%s)
    TotalSegmentator -i "$directory" -o "./dirs/segmentations/$dirname" -ta tissue_types
    python3 ./totalsegmentator/ts.get_area_of_slices.py "$dirname" --use_mean 
    python3 ./totalsegmentator/ts.get_area_of_slices.py "$dirname" --use_minmax
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    
    json_file="./ts.area_values.json"
    if [ -f "$json_file" ]; then
        jq --arg dirname "$dirname" --arg elapsed_time "$elapsed_time" '.[$dirname].elapsed_time = ($elapsed_time | tonumber)' "$json_file" > tmp.json && mv tmp.json "$json_file"
    else
        echo "JSON file not found!"
    fi

    # -----------------------------------------------------------------------------------------------------
    # Algorithm

    python3 "./fatratio/fr.dicom2niftii.py" "$dirname"

    start_time=$(date +%s) 
    python3 "./fatratio/fr.binarise.py" "$dirname" # ./dirs/output/$Dir/fat_thresh.nii.gz
    python3 "./fatratio/fr.algorithm.py" "$dirname" --use_mean # ./dirs/output/$Dir/--use_mean_combined_result.nii.gz 
    python3 "./fatratio/fr.algorithm.py"  "$dirname" --use_minmax # ./dirs/output/$Dir/--use_mean_combined_result.nii.gz 

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    json_file="./fr.area_values.json"
    if [ -f "$json_file" ]; then
        jq --arg dirname "$dirname" --arg elapsed_time "$elapsed_time" '.[$dirname].elapsed_time = ($elapsed_time | tonumber)' "$json_file" > tmp.json && mv tmp.json "$json_file"
    else
        echo "JSON file not found!"
    fi

    # -----------------------------------------------------------------------------------------------------
    # Merge Excel files

    python3 ./generic/mergejsons.py "$dirname"

    # -----------------------------------------------------------------------------------------------------
    # Compare segmentations via Units and all...

    # -----------------------------------------------------------------------------------------------------
    # Ending this case and moving on to another
    echo "Processed $directory" >> ./processed.txt

    echo ' =============================='
    # pause_for_debug
    
}

# Function to suppress output
suppress_output() {
  exec 3>&1 4>&2  # Save current stdout and stderr
  exec 1>/dev/null 2>&1  # Redirect stdout and stderr to /dev/null
}

# Function to enable output
enable_output() {
  exec 1>&3 2>&4  # Restore stdout and stderr
  exec 3>&- 4>&-  # Close temporary file descriptors
}

# Function to add a breakpoint
pause_for_debug() {
    echo "$1"
    read -n 1 -s -r -p "Press any key to continue..."
    echo ""
}

# Activate Python environment
source /tbcrohn_py39/bin/activate

# suppress output
# suppress_output

# Set the superdirectory
superdirectory="/home/medicalai/cd_itb/dataset"

rm -rfv ./z_values.json
rm -rfv ./dcmhead.json
rm -rfv ./fr.area_values.json
rm -rfv ./ts.area_values.json
rm -rfv ./processed.txt

# Process directories in each category
# enable_output
echo 'Working on iTB'
# suppress_output
for dirpath in "$superdirectory/itb"/*/; do
    if [ -d "$dirpath" ]; then
        echo "$dirpath"
        process_directory "$dirpath" "iTB"
        # enable_output
        echo "Processed $dirpath"
        # suppress_output
        
    fi
done

# enable_output
echo 'Working on Crohns:'
# suppress_output
for dirpath in "$superdirectory/crohn"/*/; do
    if [ -d "$dirpath" ]; then
        echo "$dirpath"
        process_directory "$dirpath" "CD"
        # enable_output
        echo "Processed $dirpath"
        # suppress_output

    fi
done

# enable_output
echo 'Working on Normal'
# suppress_output
for dirpath in "$superdirectory/normal"/*/; do
    if [ -d "$dirpath" ]; then
        echo "$dirpath"
        process_directory "$dirpath" "N"
        # enable_output
        echo "Processed $dirpath"
        # suppress_output
    fi
done

# enable_output
echo 'done with generating z ranges'