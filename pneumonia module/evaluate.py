# import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# import torch.nn as nn
# from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
# import matplotlib.pyplot as plt
# import numpy as np

# def evaluate_model(model_name, data_dir, split='test'):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])

#     dataset = datasets.ImageFolder(data_dir, transform=transform)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

#     if model_name == "densenet201":
#         model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
#         model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
#     else:
#         raise ValueError("Model not supported")

#     model.load_state_dict(torch.load(f"{model_name}_model.pth"))
#     model = model.to(device)  # Move the model to GPU if available
#     model.eval()

#     y_true = []
#     y_pred = []
#     y_scores = []

#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())
#             y_scores.extend(outputs.cpu().numpy()[:, 1])

#     print(f"{split.capitalize()} Classification Report:")
#     print(classification_report(y_true, y_pred))

#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     auc_score = auc(recall, precision)
#     roc_auc = roc_auc_score(y_true, y_scores)

#     print(f"{split.capitalize()} AUC Score: {auc_score}")
#     print(f"{split.capitalize()} ROC AUC Score: {roc_auc}")

#     plt.figure()
#     plt.plot(recall, precision, marker='.', label=f'PR curve (AUC = {auc_score:.2f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend()
#     plt.title(f'{split.capitalize()} Precision-Recall Curve')
#     plt.savefig(f"{model_name}_{split}_pr_curve.png")

# if __name__ == "__main__":
#     import sys
#     model_name = sys.argv[1]
#     data_dir = 'temp/val'
#     evaluate_model(model_name, data_dir, split='test')

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model_name, data_dir, split='test'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    if model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise ValueError("Model not supported")

    model.load_state_dict(torch.load(f"{model_name}_model.pth", map_location=torch.device('cpu')))
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy()[:, 1])

    print(f"{split.capitalize()} Classification Report:")
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    balanced_accuracy = (precision + recall) / 2

    print(f"{split.capitalize()} Precision: {precision}")
    print(f"{split.capitalize()} Recall: {recall}")
    print(f"{split.capitalize()} Balanced Accuracy (Precision+Recall)/2: {balanced_accuracy}")

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    auc_score = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_true, y_scores)

    print(f"{split.capitalize()} AUC Score: {auc_score}")
    print(f"{split.capitalize()} ROC AUC Score: {roc_auc}")

    plt.figure()
    plt.plot(recall_curve, precision_curve, marker='.', label=f'PR curve (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(f'{split.capitalize()} Precision-Recall Curve')
    plt.savefig(f"{model_name}_{split}_pr_curve.png")

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    data_dir = 'temp/val'
    evaluate_model(model_name, data_dir, split='test')