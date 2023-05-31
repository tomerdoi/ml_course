import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


# Define the flower dataset class
class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Load and transform the dataset
root_dir = '/Users/tomerdoitshman/Desktop/other/D_non_shared'  # Replace with the path to the extracted dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to the desired input size for the ResNet model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])
dataset = FlowerDataset(root_dir, transform=transform)

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load the pre-trained ResNet model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

# Replace the last linear layer for classification
num_classes = 102  # Number of flower classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
