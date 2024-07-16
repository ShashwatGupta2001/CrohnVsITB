import os
import pydicom
import numpy as np
import dicom2nifti
import zipfile
import nibabel as nib
import sys
import dicom2nifti
import json
import os

def sort_dicom_files(dicom_files):
    """Sort the DICOM files by the numerical part of the file name, ignoring the 'I' prefix."""
    dicom_files.sort(key=lambda f: int(os.path.basename(f).split('I')[1].split('.')[0]))
    return dicom_files

def preprocess_dicom_files(dicom_files, verbose=False):
    """Preprocess DICOM files to set RescaleIntercept to 0."""
    for dicom_file in dicom_files:
        ds = pydicom.dcmread(dicom_file)
        
        if 'RescaleIntercept' in ds:
            ds.RescaleIntercept = 0
        
        # Save the modified DICOM file
        ds.save_as(dicom_file)

    if verbose:
        print(f"Set RescaleIntercept to 0 for all dicoms")

def postprocess_dicom_files(dicom_files, verbose=False):
    """Preprocess DICOM files to set RescaleIntercept to 0."""
    for dicom_file in dicom_files:
        ds = pydicom.dcmread(dicom_file)
        
        if 'RescaleIntercept' in ds:
            ds.RescaleIntercept = -1024

        # Save the modified DICOM file
        ds.save_as(dicom_file)

    if verbose:
        print(f"Set RescaleIntercept to -1024 for all dicoms")
        

def reset_rescale_intercept_in_nifti(nifti_file, intercept_value, verbose=False):
    """Reset RescaleIntercept in the NIfTI file header."""
    nifti_img = nib.load(nifti_file)
    hdr = nifti_img.header

    # Adjusting the header, assuming srow_z contains the intercept.
    if 'srow_z' in hdr:
        hdr['srow_z'][3] = intercept_value
        if verbose:
            print(f"Reset RescaleIntercept to {intercept_value} in NIfTI header for file: {nifti_file}")
    
    # Save the modified NIfTI file
    nib.save(nifti_img, nifti_file)


def dcm_to_nifti(dicom_folder, output_path, tmp_dir=None, verbose=False):
    """
    Uses pydicom and nibabel to convert DICOM series to NIfTI (also works on windows)

    input_path: a directory of dicom slices or a zip file of dicom slices
    output_path: a nifti file path
    tmp_dir: extract zip file to this directory, else to the same directory as the zip file
    """
    input_path = f"./dirs/segmentorviewer/{dicom_folder}"
    # Check if input_path is a zip file and extract it
    if zipfile.is_zipfile(input_path):
        if verbose:
            print(f"Extracting zip file: {input_path}")
        
        extract_dir = os.path.splitext(input_path)[0] if tmp_dir is None else os.path.join(tmp_dir, "extracted_dcm")
        
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
        
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            input_path = extract_dir

    if verbose:
        print(f"Reading DICOM files from {input_path}")
        
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preprocess DICOM files to set RescaleIntercept to 0
    dicom_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.dcm')]
    preprocess_dicom_files(dicom_files, verbose=verbose)
    
    try:
        dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=True)
        if verbose:
            print(f"Conversion successful. NIfTI file saved to {output_path}")
    except Exception as e:
        print(f"An error occurred during DICOM to NIfTI conversion: {e}")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        print(f"Temporary directory: {tmp_dir}")
    finally:
       postprocess_dicom_files(dicom_files, verbose=verbose) 

    # Reset RescaleIntercept in NIfTI header to -1024
    # reset_rescale_intercept_in_nifti(output_path, -1024, verbose=verbose)

# def get_headers(dicom_folder):
#     dicom_input = dicom2nifti.common.read_dicom_directory(dicom_folder)
#     header = dicom_input[0]

#     # Define the coordinates for the required headers
#     coordinates = {
#         "Pitch": (0x01f1, 0x1026),
#         "Acquisition Type": [(0x0018, 0x9302), (0x01f1, 0x1001)],
#         "Acquisition Length": (0x01f1, 0x1008),
#         "Acquisition Duration": (0x00e1, 0x1050),
#         "Number of Study Related Instances": (0x0020, 0x1208),
#         "Manufacturer": (0x0008, 0x0070),
#         "Patient's Sex": (0x0010, 0x0040),
#         "Contrast/Bolus Agent": (0x0018, 0x0010)
#     }

#     # Retrieve and print the values based on the coordinates
#     for header, coord in coordinates.items():
#         if isinstance(coord, tuple):  # Single coordinate
#             value = dicom_input[0].get(coord)
#             print(f"{header}: {value}")
#         elif isinstance(coord, list):  # Multiple coordinates
#             for c in coord:
#                 value = dicom_input[0].get(c)
#                 if value is not None:
#                     print(f"{header}: {value}")
#                     break
    
#     if 'Manufacturer' not in header or 'Modality' not in header:
#         print('Manufacturer or Modality not found in header')
#     else:
#         print("Machine is:", header.Manufacturer.upper())
#         print("Header is:", header)
def process_string(input_string):
    if input_string is None:
        return input_string

    if ':' in input_string:
        result = input_string.split(':', 1)[1]
        result = result.strip().strip(',"\'')
        return result
    
    return input_string

def get_headers(dicom_folder, json_file_path):
    dicom_input = dicom2nifti.common.read_dicom_directory(dicom_folder)
    header = dicom_input[0]

    # Define the coordinates for the required headers
    coordinates = {
        "Pitch": (0x01f1, 0x1026),
        "Acquisition Type": [(0x0018, 0x9302), (0x01f1, 0x1001)],
        "Acquisition Length": (0x01f1, 0x1008),
        "Acquisition Duration": (0x00e1, 0x1050),
        "Number of Study Related Instances": (0x0020, 0x1208),
        "Manufacturer": (0x0008, 0x0070),
        "Patient's Sex": (0x0010, 0x0040),
        "Contrast/Bolus Agent": (0x0018, 0x0010)
    }

    # Retrieve the values based on the coordinates
    result = {}
    for header_name, coord in coordinates.items():
        value = None
        if isinstance(coord, tuple):  # Single coordinate
            value = header.get(coord)
        elif isinstance(coord, list):  # Multiple coordinates
            for c in coord:
                value = header.get(c)
                if value is not None:
                    break
        result[header_name] = process_string(str(value)) if value is not None else None

    # Load existing data from JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    # Get directory name as the key
    directory = os.path.basename(dicom_folder)

    # Update or add the new result
    if directory in existing_data:
        existing_data[directory].update(result)
    else:
        existing_data[directory] = result

    # Save updated data to JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

if __name__ == "__main__":
    dicom_folder = sys.argv[1]
    print(dicom_folder)
    output_dir = f'./dirs/output/{dicom_folder}'
    output_file = 'output.nii.gz'
    tmp_dir = './dirs/tmp'

    json_file_path = './dcmhead.json'
    get_headers(f"./dirs/segmentorviewer/{dicom_folder}", json_file_path)

    dcm_to_nifti(dicom_folder, os.path.join(output_dir, output_file), tmp_dir, verbose=True)

