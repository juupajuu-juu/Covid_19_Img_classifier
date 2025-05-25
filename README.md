# Covid_19_Img_classifier
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
train_dir = r"C:\Users\Downloads\archive\Covid19-dataset\train" # <- use your own path here
test_dir = r"C:\Users\Downloads\archive\Covid19-dataset\test" # <- use your own path here

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes
model = model.to(device)
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # or more
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {running_loss:.4f}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Select a test image
img_path = r"C:\Users\Downloads\archive\Covid19-dataset\test\Covid\098.jpeg"  # <- CHANGE THIS TO A REAL IMAGE PATH

# Preprocess the image
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# Set up Grad-CAM hooks
gradients = []
activations = []
target_layer = model.layer4[1].conv2

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Register hooks
handle_f = target_layer.register_forward_hook(forward_hook)
handle_b = target_layer.register_backward_hook(backward_hook)

# Forward pass
output = model(input_tensor)
pred_class = output.argmax().item()
print(f"Predicted class: {pred_class} ({train_dataset.classes[pred_class]})")

# Backward pass
model.zero_grad()
output[0, pred_class].backward()

# Generate Grad-CAM
acts = activations[0].squeeze().cpu().detach()
grads = gradients[0].squeeze().cpu().detach()
weights = grads.mean(dim=(1, 2))
cam = (weights[:, None, None] * acts).sum(dim=0)
cam = torch.relu(cam)
cam = cam - cam.min()
cam = cam / cam.max()
cam = cam.numpy()
cam = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# Overlay heatmap on original image
original = np.array(img.resize((224, 224)))
overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

# Show results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(overlay)
plt.axis('off')
plt.show()

# Remove hooks
handle_f.remove()
handle_b.remove()

