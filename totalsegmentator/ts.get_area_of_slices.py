import os
import pydicom
import numpy as np
import dicom2nifti
from dicom2nifti import common
import zipfile
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys

def readjson(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def use_custom(s, e, directory, option):
    s = int(s)
    e = int(e)
    print(f"Using custom normalization with arguments: {s}, {e}")

    # Load the NIfTI file
    nii_file_path = f'./dirs/segmentations/{directory}/subcutaneous_fat.nii.gz'  # Replace with the correct path to your file
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()

    # Compute the volume of the white area in the specified slices
    subcut_volume = np.sum(nii_data[:, :, s:e+1] == 1)  # Assuming white area is labeled as 1
    total_slice_volume = np.prod(nii_data[:, :, s:e+1].shape)

    # Load the NIfTI file
    nii_file_path = f'./dirs/segmentations/{directory}/torso_fat.nii.gz'  # Replace with the correct path to your file
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()

    # Compute the volume of the white area in the specified slices
    visceral_volume = np.sum(nii_data[:, :, s:e+1] == 1)  # Assuming white area is labeled as 1
    total_slice_volume = np.prod(nii_data[:, :, s:e+1].shape)

    print(f"Subcutaneious volume: {subcut_volume}")
    print(f"visceral volume: {visceral_volume}")
    print(f"Total slice volume: {total_slice_volume}")

    # Load existing JSON data
    json_file_path = './ts.area_values.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    result = {
        option + "_sv": int(subcut_volume),
        option + "_vv": int(visceral_volume),
        option + "_tsv": int(total_slice_volume),
        option + "_ratio":str(visceral_volume/subcut_volume)
    }

    # Update the existing_data dictionary with the new result
    if directory in existing_data:
        existing_data[directory].update(result)
    else:
        existing_data[directory] = result

    # Save the updated JSON data to a file
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    print(f"JSON result saved to {json_file_path}")

def main():
    if len(sys.argv) < 3:
        print("Invalid number of arguments. Usage: <script> <directory> <option> [<arg1> <arg2>]")
        return

    directory = sys.argv[1]

    json_value = readjson('./z_values.json')
    min_value = json_value[directory]['min']
    max_value = json_value[directory]['max']
    mean_value = json_value[directory]['mean']

    option = sys.argv[2]

    if option == '--use_mean':
        use_custom(mean_value, mean_value, directory, "use_mean")
    elif option == '--use_minmax':
        use_custom(min_value, max_value, directory, "use_minmax")
    elif option == '--use_custom':
        if len(sys.argv) != 5:
            print("Invalid number of arguments for --use_custom. Expected 2 arguments.")
            return
        arg1 = sys.argv[3]
        arg2 = sys.argv[4]
        use_custom(arg1, arg2, directory, "use_custom")
    else:
        print(f"Unknown option: {option}")

if __name__ == "__main__":
    main()
