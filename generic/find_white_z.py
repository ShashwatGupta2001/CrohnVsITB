# import nibabel as nib
# import numpy as np
# import json
# import os

# def process_nifti(nifti_file_path, json_file_path):
#     # Load the NIfTI file
#     nifti_img = nib.load(nifti_file_path)
#     nifti_data = nifti_img.get_fdata()

#     # Assuming white color has the maximum intensity value, you can use a threshold or the maximum value directly
#     white_value = np.max(nifti_data)

#     # Find the z-slices containing the white object
#     z_slices = []

#     for z in range(nifti_data.shape[2]):
#         if white_value in nifti_data[:, :, z]:
#             z_slices.append(z)

#     # Calculate min, max, and mean
#     if z_slices:
#         min_z = int(np.min(z_slices))
#         max_z = int(np.max(z_slices))
#         mean_z = int(np.mean(z_slices))
#     else:
#         min_z = max_z = mean_z = None

#     # Create a dictionary to store the results
#     result = {
#         "min": min_z,
#         "max": max_z,
#         "mean": mean_z
#     }

#     # Read existing JSON content if the file exists
#     if os.path.exists(json_file_path):
#         with open(json_file_path, 'r') as json_file:
#             data = json.load(json_file)
#     else:
#         data = {}

#     # Use the filename (without path) as the key
#     filename = os.path.basename(nifti_file_path)
#     data[filename] = result

#     # Convert the dictionary to a JSON string
#     result_json = json.dumps(data, indent=4)

#     # Save the JSON string to a file
#     with open(json_file_path, 'w') as json_file:
#         json_file.write(result_json)

#     print(f"JSON result saved to {json_file_path}")

# # Example usage
# nifti_file_path = './dirs/output/combined_binary_mask.nii.gz'  # Replace with your file path
# json_file_path = './z_values.json'  # Replace with your desired JSON file path
# process_nifti(nifti_file_path, json_file_path)




import nibabel as nib
import numpy as np
import json
import os
import sys

def process_nifti(nifti_file_path, json_file_path, filename_key,elapsed_time,disease):
    # Load the NIfTI file
    nifti_img = nib.load(nifti_file_path)
    nifti_data = nifti_img.get_fdata()

    # Assuming white color has the maximum intensity value, you can use a threshold or the maximum value directly
    white_value = np.max(nifti_data)

    # Find the z-slices containing the white object
    z_slices = []

    for z in range(nifti_data.shape[2]):
        if white_value in nifti_data[:, :, z]:
            z_slices.append(z)

    # Calculate min, max, and mean
    if z_slices:
        min_z = int(np.min(z_slices))
        max_z = int(np.max(z_slices))
        mean_z = int(np.mean(z_slices))
    else:
        min_z = max_z = mean_z = None

    # Create a dictionary to store the results
    result = {
        "min": min_z,
        "max": max_z,
        "mean": mean_z,
        "time": elapsed_time,
        "disease": disease,

    }

    # Read existing JSON content if the file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {}

    # Use the provided filename_key as the key
    data[filename_key] = result

    # Convert the dictionary to a JSON string
    result_json = json.dumps(data, indent=4)

    # Save the JSON string to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(result_json)

    print(f"JSON result saved to {json_file_path}")

# Example usage with command-line arguments
if __name__ == "__main__":
    nifti_file_path = './dirs/output/combined_binary_mask.nii.gz'
    json_file_path = './z_values.json'
    filename_key = sys.argv[1]
    elapsed_time = sys.argv[2]
    disease=sys.argv[3]
    process_nifti(nifti_file_path, json_file_path, filename_key,elapsed_time,disease)