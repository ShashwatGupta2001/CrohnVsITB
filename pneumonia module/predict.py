import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import csv
import sys
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from PIL import Image
import os
import csv
import sys

def load_model(model_name, model_path):
    if model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise ValueError("Model not supported")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_path, transform):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

def process_directory(model, directory, transform, output_csv):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, file)
                prediction = predict(model, img_path, transform)
                relative_path = os.path.relpath(img_path, directory)
                results.append((relative_path, prediction))

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image', 'Prediction'])
        csvwriter.writerows(results)

if __name__ == "__main__":
    model_name = sys.argv[1]
    model_path = f"{model_name}_model.pth"
    directory = sys.argv[2]
    output_csv = sys.argv[3]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model = load_model(model_name, model_path)
    process_directory(model, directory, transform, output_csv)
