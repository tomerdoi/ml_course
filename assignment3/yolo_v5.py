import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import scipy.io as sio
from sklearn.metrics import accuracy_score, precision_score


# Define the flower dataset class
class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.labels = self.load_labels(root_dir)

    def load_labels(self, root_dir):
        labels_file = os.path.join(root_dir, 'imagelabels.mat')
        labels = sio.loadmat(labels_file)['labels'][0]
        return labels - 1  # Adjust labels to be zero-indexed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Discard the original label
        label = self.labels[idx]
        return image, label


# Load and transform the dataset
root_dir = '/Users/tomerdoitshman/Desktop/other/D_non_shared/ass3_dataset'  # Replace with the
# path to the extracted dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to the desired input size for the ResNet model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])
dataset = FlowerDataset(root_dir, transform=transform)

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

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
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    batch_idx = 0
    for images, labels in dataloader:
        print('Running on batch %d' % batch_idx)
        batch_idx += 1
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx == 1:  # Calculate accuracy and precision after the first batch
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

            accuracy = accuracy_score(labels, predicted)
            precision = precision_score(labels, predicted, average='weighted')

            print('Accuracy:', accuracy)
            print('Precision:', precision)
        break

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
