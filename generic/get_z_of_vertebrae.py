import os
import pydicom
import numpy as np
import dicom2nifti
from dicom2nifti import common
import zipfile
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

if __name__ == "__main__":
    
    directory=sys.argv[1]

    # Directory containing the segmentation files
    segmentations_dir = f'./dirs/segmentations/{directory}'

    # Get list of all segmentation files in the directory
    segmentation_files = [
        'vertebrae_L4',
        # 'subcutaneous_fat' # available through licence
        ]
    # [f for f in os.listdir(segmentations_dir) if f.endswith('.nii.gz')]

    # Load the first mask to get the shape
    first_mask = nib.load(os.path.join(segmentations_dir, segmentation_files[0]+'.nii.gz'))
    shape = first_mask.shape

    # Initialize an empty 3D array for the binary mask
    combined_mask = np.zeros(shape, dtype=bool)

    # Process each segmentation file
    for file_name in tqdm(segmentation_files):
        file_path = os.path.join(segmentations_dir, file_name+'.nii.gz')
        mask = nib.load(file_path).get_fdata()
        mask = mask.astype(bool)  # Ensure mask is boolean
        np.logical_or(combined_mask, mask, out=combined_mask)  # Combine masks with logical OR

    # Convert boolean mask to uint8 for visualization (0 for black, 255 for white)
    colored_mask = combined_mask.astype(np.uint8) * 255

    # Create a new Nifti image
    colored_mask_nii = nib.Nifti1Image(colored_mask, affine=first_mask.affine)

    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    # Save the new Nifti image
    outputdir='./dirs/output'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        print(f"Directory '{outputdir}' created")
    else:
        print(f"{outputdir} already exists, ammending it")
    nib.save(colored_mask_nii, os.path.join(outputdir,'combined_binary_mask.nii.gz'))

    print("Combined binary mask saved as 'combined_binary_mask.nii.gz'")