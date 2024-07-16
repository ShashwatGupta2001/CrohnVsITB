import os
import shutil
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def process_and_copy(src_files, dest_dir):
    for file_path in tqdm(src_files, desc=f"Processing {dest_dir}"):
        try:
            with Image.open(file_path) as img:
                grayscale_img = img.convert('L')
                # Ensure the destination directory exists
                os.makedirs(dest_dir, exist_ok=True)
                # Save the processed image in the destination directory
                dest_path = os.path.join(dest_dir, os.path.basename(file_path))
                grayscale_img.save(dest_path)
        except UnidentifiedImageError:
            print(f"Skipping file {file_path} - cannot identify image file")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def split_and_preprocess(data_dir, temp_dir, val_size=0.05):
    # Paths
    diseased_dir = os.path.join(data_dir, 'diseased')
    normal_dir = os.path.join(data_dir, 'normal')
    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'val')

    # Gather files
    diseased_files = [os.path.join(diseased_dir, f) for f in os.listdir(diseased_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Split data
    train_diseased, val_diseased = train_test_split(diseased_files, test_size=val_size, random_state=42)
    train_normal, val_normal = train_test_split(normal_files, test_size=val_size, random_state=42)

    # Process and copy files
    process_and_copy(train_diseased, os.path.join(train_dir, 'diseased'))
    process_and_copy(val_diseased, os.path.join(val_dir, 'diseased'))
    process_and_copy(train_normal, os.path.join(train_dir, 'normal'))
    process_and_copy(val_normal, os.path.join(val_dir, 'normal'))

if __name__ == "__main__":
    data_dir = 'data'
    temp_dir = 'temp'
    split_and_preprocess(data_dir, temp_dir)
