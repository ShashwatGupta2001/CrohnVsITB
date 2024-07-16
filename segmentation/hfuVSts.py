import sys
import json
import numpy as np
import nibabel as nib
import cv2
import os

def read_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def logicalor(array1, array2):
    return np.logical_or(array1, array2)

def getDice(data1, data2):
    intersection = np.logical_and(data1, data2)
    sum_data1 = data1.sum()
    sum_data2 = data2.sum()
    if sum_data1 + sum_data2 == 0:
        return 1.0 if intersection.sum() == 0 else 0.0
    dice_score = 2 * intersection.sum() / (sum_data1 + sum_data2)
    return dice_score

def getJaccard(data1, data2):
    intersection = np.logical_and(data1, data2)
    union = np.logical_or(data1, data2)
    sum_union = union.sum()
    if sum_union == 0:
        return 1.0 if intersection.sum() == 0 else 0.0
    jaccard_index = intersection.sum() / sum_union
    return jaccard_index

def save_as_nifti(data, reference_img, output_path):
    new_img = nib.Nifti1Image(data.astype(np.uint8), reference_img.affine, reference_img.header)
    nib.save(new_img, output_path)

# Corrected the function
def func(slice_img):
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(cv2.erode(slice_img, kernel, iterations=2), kernel, iterations=2)
    if len(thresh.shape) == 3:  # Check if image has multiple channels
        img = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    else:
        img = thresh
    return img

def main():
    if len(sys.argv) < 2:
        print("No arguments provided.")
        return

    directory = sys.argv[1]
    nifti_file = f'./dirs/output/{directory}/fat_thresh.nii.gz'
    json_value = read_json('./z_values.json')

    img = nib.load(nifti_file)
    data1 = img.get_fdata()

    # Process each slice in data1 (cut along z-axis)
    processed_slices = []
    for i in range(data1.shape[2]):
        slice_img = data1[:, :, i]
        processed_slice = func(slice_img)
        processed_slices.append(processed_slice)
    
    # Reconstruct data1 from processed slices
    data1 = np.stack(processed_slices, axis=2)
    data1 = data1 != 0

    nifti_file1 = f'./dirs/segmentations/{directory}/subcutaneous_fat.nii.gz'
    img1 = nib.load(nifti_file1)
    data21 = img1.get_fdata()
    data21 = data21 != 0
    
    nifti_file2 = f'./dirs/segmentations/{directory}/torso_fat.nii.gz'
    img2 = nib.load(nifti_file2)
    data22 = img2.get_fdata()
    data22 = data22 != 0
     
    data2 = logicalor(data21, data22)

    min_value = json_value[directory]['min']
    max_value = json_value[directory]['max']
    mean_value = json_value[directory]['mean']

    option = sys.argv[2]
    if option == '--use_mean':
        data1 = data1[:, :, mean_value:mean_value+1]
        data2 = data2[:, :, mean_value:mean_value+1]
    elif option == '--use_minmax':
        data1 = data1[:, :, min_value:max_value + 1]
        data2 = data2[:, :, min_value:max_value + 1]
    elif option == '--use_custom':
        if len(sys.argv) != 5:
            print("Invalid number of arguments for --use_custom. Expected 2 arguments.")
            return
        arg1 = int(sys.argv[3])
        arg2 = int(sys.argv[4])
        data1 = data1[:, :, arg1:arg2 + 1]
        data2 = data2[:, :, arg1:arg2 + 1]
    else:
        print(f"Unknown option: {option}")
        return

    # Debug prints to check the contents of the slices
    # print("Data1 shape:", data1.shape)
    # print("Data2 shape:", data2.shape)
    # print("Data1 sum:", data1.sum())
    # print("Data2 sum:", data2.sum())

    if data1.sum() == 0 or data2.sum() == 0:
        print("One of the data arrays is empty after slicing.")

    # Save data1 and data2 as nii.gz files
    output_path1 = f'./dirs/output/{directory}/processed_data1.nii.gz'
    output_path2 = f'./dirs/output/{directory}/processed_data2.nii.gz'
    save_as_nifti(data1, img, output_path1)
    save_as_nifti(data2, img, output_path2)
    
    dice_score = getDice(data1, data2)
    jaccard_score = getJaccard(data1, data2)

    # print("DICE: ", dice_score)
    # print("Jaccard: ", jaccard_score)

    # Load existing JSON data
    json_file_path = './seg.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    result = {
        "hfuVts_" + option + "_dice": str(dice_score),
        "hfuVts_" + option + "_jaccard": str(jaccard_score),
    }

    # Update the existing_data dictionary with the new result
    if directory in existing_data:
        existing_data[directory].update(result)
    else:
        existing_data[directory] = result

    # Save the updated JSON data to a file
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    # print(f"JSON result saved to {json_file_path}")

if __name__ == "__main__":
    main()



