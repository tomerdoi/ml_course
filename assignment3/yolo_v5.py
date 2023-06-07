import torch
import torch.nn as nn
from data_loader import FlowerDataLoader
from sklearn.metrics import accuracy_score, precision_score


class YoloV5:
    def __init__(self):
        self.train_folder = '/Users/tomerdoitshman/Desktop/other/D_non_shared/ass3_dataset'
        self.flowers_data_loader = FlowerDataLoader(root_dir=self.train_folder, batch_size=100)
        self.dataloader = None
        self.model = None

    def load_data(self):
        try:
            self.dataloader = self.flowers_data_loader.get_train_data_loader()
        except Exception as e:
            print('Exception %s occurred during load_data.' % e)

    def load_model(self):
        try:
            # Load the pre-trained ResNet model
            self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        except Exception as e:
            print('Exception %s occurred during load_model.' % e)

    def train_model(self):
        try:
            # Replace the last linear layer for classification
            num_classes = 102  # Number of flower classes
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            # Training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters())
            # Training loop
            num_epochs = 1
            for epoch in range(num_epochs):
                running_loss = 0.0
                batch_idx = 0
                for images, labels in self.dataloader:
                    print('Running on batch %d' % batch_idx)
                    batch_idx += 1
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.cpu().numpy()
                    labels = labels.cpu().numpy()
                    accuracy = accuracy_score(labels, predicted)
                    precision = precision_score(labels, predicted, average='weighted')
                    print('Accuracy:', accuracy)
                    print('Precision:', precision)
                epoch_loss = running_loss / len(self.dataloader)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        except Exception as e:
            print('Exception %s occurred during train_model.' % e)


if __name__ == '__main__':
    yolo_v5 = YoloV5()
    yolo_v5.load_data()
    yolo_v5.load_model()
    yolo_v5.train_model()
