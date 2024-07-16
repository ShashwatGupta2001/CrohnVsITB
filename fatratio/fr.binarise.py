import nibabel as nib
import numpy as np
import sys

def segment_ct_scan(input_file, output_file, hu_min, hu_max):
    # Load the NIfTI file
    nii = nib.load(input_file)
    data = nii.get_fdata()

    # Segment the data based on the Hounsfield unit range
    segmented_data = np.where((data >= hu_min) & (data <= hu_max), 1, 0)

    # Create a new NIfTI image with the segmented data
    segmented_nii = nib.Nifti1Image(segmented_data, nii.affine, nii.header)

    # Save the new NIfTI file
    nib.save(segmented_nii, output_file)

# Example usage
dicom_folder = sys.argv[1]
input_file = f'./dirs/output/{dicom_folder}/output.nii.gz'  # Path to your input NIfTI file
output_file = f'./dirs/output/{dicom_folder}/fat_thresh.nii.gz'  # Path to save the segmented NIfTI file
hu_min = 850  # Minimum HU value of the range
hu_max = 974  # Maximum HU value of the range

segment_ct_scan(input_file, output_file, hu_min, hu_max)
