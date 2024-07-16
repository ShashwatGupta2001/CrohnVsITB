# import torch
# import torch.nn.functional as F
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import cv2
# import os

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None

#         target_layer.register_forward_hook(self.save_activation)
#         target_layer.register_backward_hook(self.save_gradient)

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def save_activation(self, module, input, output):
#         self.activations = output

#     def __call__(self, x):
#         self.model.eval()
#         output = self.model(x)

#         target = output.max(1)[1].item()
#         output[:, target].backward()
        
#         gradients = self.gradients.cpu().data.numpy()[0]
#         activations = self.activations.cpu().data.numpy()[0]

#         weights = np.mean(gradients, axis=(1, 2))
#         cam = np.zeros(activations.shape[1:], dtype=np.float32)

#         for i, w in enumerate(weights):
#             cam += w * activations[i]

#         cam = np.maximum(cam, 0)
#         cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
#         cam = cam - np.min(cam)
#         cam = cam / np.max(cam)

#         return cam

# def apply_gradcam(image_path, model, gradcam, output_dir):
#     image = Image.open(image_path).convert('L')
#     image = image.resize((224, 224))
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#     ])
#     input_image = transform(image).unsqueeze(0).to(device)

#     cam = gradcam(input_image)

#     image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
#     cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     cam = np.float32(cam) + np.float32(image)
#     cam = 255 * cam / np.max(cam)
#     cam = np.uint8(cam)

#     output_path = os.path.join(output_dir, f"gradcam.{os.path.basename(image_path)}")
#     cv2.imwrite(output_path, cam)
#     print(f"Saved Grad-CAM to {output_path}")

# if __name__ == "__main__":
#     import sys
#     model_name = sys.argv[1]
#     image_path = sys.argv[2]
#     output_dir = sys.argv[3]

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     if model_name == "densenet201":
#         model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
#         model.features.conv0 = torch.nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
#         model = model.to(device)
#         target_layer = model.features.denseblock4.denselayer16.conv2
#     else:
#         raise ValueError("Model not supported")

#     gradcam = GradCAM(model, target_layer)
#     apply_gradcam(image_path, model, gradcam, output_dir)

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def __call__(self, x):
        self.model.eval()
        output = self.model(x)

        target = output.max(1)[1].item()
        output[:, target].backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

def apply_gradcam(image_path, model, gradcam, output_dir):
    image = Image.open(image_path).convert('L')
    image = image.resize((224, 224))
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    cam = gradcam(input_image)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, cam)
    print(f"Saved Grad-CAM to {output_path}")

def process_directory(image_dir, model, gradcam, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                apply_gradcam(image_path, model, gradcam, output_dir)

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    image_dir = sys.argv[2]
    output_dir = sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.features.conv0 = torch.nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(device)
        target_layer = model.features.denseblock4.denselayer16.conv2
    else:
        raise ValueError("Model not supported")

    gradcam = GradCAM(model, target_layer)
    process_directory(image_dir, model, gradcam, output_dir)
