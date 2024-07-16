import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from tqdm import tqdm
import json

def get_weights_for_balanced_classes(dataset):
    class_counts = [0, 0]
    for _, label in dataset:
        class_counts[label] += 1

    num_samples = sum(class_counts)
    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[label] for _, label in dataset]

    return weights

def train_model(model_name, train_dir, epochs=4, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    weights = get_weights_for_balanced_classes(train_dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    if model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise ValueError("Model not supported")

    model = model.to(device)  # Move the model to GPU if available

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_log = {'loss': [], 'accuracy': []}
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        training_log['loss'].append(epoch_loss)
        training_log['accuracy'].append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")

    with open(f"{model_name}_training_log.json", 'w') as f:
        json.dump(training_log, f)

    torch.save(model.state_dict(), f"{model_name}_model.pth")

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    train_dir = 'temp/train'
    train_model(model_name, train_dir)
