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
            # Replace the last linear layer for classification
            num_classes = 102  # Number of flower classes
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        except Exception as e:
            print('Exception %s occurred during load_model.' % e)

    def train_model(self):
        try:
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

    def check_model(self, validate=True):
        try:
            # Get the validation data loader
            if validate:
                phase_name = 'Validation'
                data_loader = self.flowers_data_loader.get_validation_data_loader()
            else:
                phase_name = 'Test'
                data_loader = self.flowers_data_loader.get_test_data_loader()
            # Set the model to evaluation mode
            self.model.eval()
            # Lists to store the predicted labels and true labels
            predicted_labels = []
            true_labels = []
            # Iterate over the validation data
            for images, labels in data_loader:
                # Move the data to the GPU, if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass to obtain the predictions
                with torch.no_grad():
                    outputs = self.model(images)
                # Get the predicted labels
                _, predicted = torch.max(outputs.data, 1)
                # Append the predicted and true labels to the respective lists
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            # Calculate the validation accuracy and precision
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='weighted')
            # Print the validation results
            print(f"{phase_name} Accuracy: {accuracy:.4f}")
            print(f"{phase_name} Precision: {precision:.4f}")
        except Exception as e:
            print('Exception %s occurred during check_model.' % e)


if __name__ == '__main__':
    yolo_v5 = YoloV5()
    yolo_v5.load_data()
    yolo_v5.load_model()
    yolo_v5.train_model()
    yolo_v5.check_model(validate=True)
    yolo_v5.check_model(validate=False)
