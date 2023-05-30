from vgg_pytorch import VGG
model = VGG.from_pretrained('vgg19', num_classes=10)

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from vgg_pytorch import VGG


num_epochs = 1000
# Set the path to the train and test folders
train_folder = '/Users/tomerdoitshman/Desktop/other/D_non_shared'
test_folder = '/Users/tomerdoitshman/Desktop/other/D_non_shared'

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to VGG input size
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensors
])

# Create train and test datasets
train_dataset = ImageFolder(train_folder, transform=transform)
test_dataset = ImageFolder(test_folder, transform=transform)

# Create train and test data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pre-trained VGG model
model = VGG.from_pretrained('vgg19', num_classes=102)  # Update num_classes to match your dataset

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(10):  # Adjust the number of epochs as needed
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation on test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Epoch: {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
