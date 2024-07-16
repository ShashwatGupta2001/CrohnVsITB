#!/bin/bash

model_name="densenet201"
directory="./final"
explanable_directory="./gradcam"

# -------------------------------------------------------------------------------

source lung_classification_env/bin/activate

# -------------------------------------------------------------------------------

echo "1. Splitting and Preprocessing...."
# read -n 1
python3 split_and_preprocess.py

echo "2. Training...."
# read -n 1
python3 train.py $model_name

echo "3. Evaluationg...."
# read -n 1
python3 evaluate.py $model_name

echo "4. Plotting..."
# read -n 1
python3 plot.py $model_name

echo "5. Predicting..." 
python3 predict.py $model_name $directory output_predictions.csv

echo "6. GradCAM...." 
create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "Directory $1 created."
    else
        echo "Directory $1 already exists."
    fi
}
create_directory $explanable_directory
python gradcam.py $model_name ./final/ $explanable_directory

# -------------------------------------------------------------------------------

deactivate