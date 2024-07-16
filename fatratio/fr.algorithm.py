import sys
import json
import math
import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm
import os

def line_iter(start, end):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield (x0, y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def find_last_point(img, start, end):
    last_black_point = None
    last_white_point = None
    answer = None
    for x, y in line_iter(start, end):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if img[y, x] == 0:
                last_black_point = (x, y)
            elif last_black_point and img[y, x] != 0:
                answer = last_black_point
            if img[y, x] > 0:
                last_white_point = (x, y)
    return answer, last_white_point

def compute_areas(img, granular_degree):
    key_points = []
    height, width = img.shape[:2]
    center = (int(width / 2), int(height / 2))
    subcut_area = 0
    for angle in np.arange(0, 360, granular_degree):
        theta = np.deg2rad(angle)
        end_x = int(center[0] + math.cos(theta) * width)
        end_y = int(center[1] + math.sin(theta) * height)
        bk, wh = find_last_point(img, center, (end_x, end_y))
        if bk and wh:
            subcut_area += 0.5 * ((wh[0] - center[0]) ** 2 + (wh[1] - center[1]) ** 2 - (bk[0] - center[0]) ** 2 - (bk[1] - center[1]) ** 2) * np.deg2rad(granular_degree)
            key_points.append((bk, wh))
    return subcut_area, key_points

def compute_white_area(img):
    return np.sum(img == 255)

def removal_ct(img, removal_method):
    if removal_method == 'gaussian-blur':
        blurred_image = cv2.GaussianBlur(img, (11, 11), 0)
        _, thresh = cv2.threshold(blurred_image, 155, 255, cv2.THRESH_BINARY)
        return thresh
    elif removal_method == 'median-blur':
        blurred_image = cv2.medianBlur(img, 11)
        _, thresh = cv2.threshold(blurred_image, 155, 255, cv2.THRESH_BINARY)
        return thresh
    elif removal_method == 'bilateral-filter':
        blurred_image = cv2.bilateralFilter(img, 11, 17, 17)
        _, thresh = cv2.threshold(blurred_image, 155, 255, cv2.THRESH_BINARY)
        return thresh
    elif removal_method == 'erosion-dilation':
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        return dilation
    elif removal_method == 'dilation-erosion':
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        return erosion

def single_img(slice_data, granular_degree=0.05):
    thresh = removal_ct(slice_data, removal_method='erosion-dilation')
    
    if len(thresh.shape) == 2:
        img = thresh
    else:
        img = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    subcut_area, key_points = compute_areas(img, granular_degree)
    total_area = compute_white_area(img)

    kernel = np.ones((3, 3), np.uint8)
    segmentation = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in key_points:
        cv2.line(segmentation, line[0], line[1], (255, 255, 255), 1)
    segmentation = cv2.dilate(segmentation, kernel, iterations=3)
    segmentation = cv2.erode(segmentation, kernel, iterations=3)

    return subcut_area, total_area, img.shape[0] * img.shape[1], key_points, segmentation

def read_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def save_as_png(image, filename):
    cv2.imwrite(filename, image)

def main():
    if len(sys.argv) < 3:
        print("No arguments provided.")
        return

    directory = sys.argv[1]
    nifti_file = f'./dirs/output/{directory}/fat_thresh.nii.gz'
    json_value = read_json('./z_values.json')

    min_value = json_value[directory]['min']
    max_value = json_value[directory]['max']
    mean_value = json_value[directory]['mean']

    option = sys.argv[2]
    if option == '--use_mean':
        slices_to_process = [mean_value]
    elif option == '--use_minmax':
        slices_to_process = list(range(min_value, max_value + 1))
    elif option == '--use_custom':
        if len(sys.argv) != 5:
            print("Invalid number of arguments for --use_custom. Expected 2 arguments.")
            return
        arg1 = int(sys.argv[3])
        arg2 = int(sys.argv[4])
        slices_to_process = list(range(arg1, arg2 + 1))
    else:
        print(f"Unknown option: {option}")
        return

    img = nib.load(nifti_file)
    data = img.get_fdata()

    SFA = 0
    TFA = 0
    TA = 0
    result_img_combined = None

    print("Slices_to_process : ", len(slices_to_process))

    result_img_combined = []

    granular_degree = 1
    result_img_paths = []

    for idx, i in tqdm(enumerate(slices_to_process),desc="Slices"):

        if i < 0 or i >= data.shape[2]:
            print(f"Slice index {i} is out of range. Skipping.")
            continue
        slice_data = data[:, :, i]
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
        slice_data = slice_data.astype(np.uint8)

        if len(slice_data.shape) == 3:
            slice_data = cv2.cvtColor(slice_data, cv2.COLOR_BGR2GRAY)
        else:
            slice_data = slice_data.reshape(slice_data.shape[0], slice_data.shape[1], 1)

        subcut_fat_area, total_fat_area, total_area, _, result_img_last = single_img(slice_data, granular_degree)
        SFA += subcut_fat_area
        TFA += total_fat_area
        TA += total_area

        result_img_last = cv2.cvtColor(result_img_last, cv2.COLOR_BGR2GRAY)
        result_img_combined.append(result_img_last)

    result_img_combined = np.array(result_img_combined)
    result_img_combined = np.transpose(result_img_combined, (1, 2, 0))

    json_file_path = './fr.area_values.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    result = {
        option + "_sv": int(SFA),
        option + "_vv": int(TFA - SFA),
        option + "_tsv": int(TA),
        option + "_ratio": str(1.0*TFA/SFA -1) ,
    }
    
    if directory in existing_data:
        existing_data[directory].update(result)
    else:
        existing_data[directory] = result

    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    new_img = nib.Nifti1Image(result_img_combined, img.affine)
    nib.save(new_img, f'./dirs/output/{directory}/{option}_combined_result.nii.gz')

if __name__ == "__main__":
    main()




    