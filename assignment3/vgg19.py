import torch
from vgg_pytorch import VGG
from data_loader import FlowerDataLoader
from logger_utils import LoggerUtils
from sklearn.metrics import accuracy_score, precision_score

num_epochs = 10


class VGG19:
    def __init__(self):
        self.train_folder = '/Users/tomerdoitshman/Desktop/other/D_non_shared/ass3_dataset'
        self.flowers_data_loader = FlowerDataLoader(root_dir=self.train_folder, batch_size=100)
        self.dataloader = None
        self.model = None
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='vgg19.log')

    def load_data(self):
        try:
            self.dataloader = self.flowers_data_loader.get_train_data_loader()
        except Exception as e:
            self.logger.error('Exception %s occurred during load_data.' % e)

    def load_model(self):
        try:
            # Load the pre-trained VGG model
            self.model = VGG.from_pretrained('vgg19', num_classes=102)  # Update num_classes to match your dataset
        except Exception as e:
            self.logger.error('Exception %s occurred during load_model.' % e)

    def train_model(self):
        try:
            # Define the loss function and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            # Training loop
            for epoch in range(num_epochs):
                self.logger.info('Running on epoch %d.' % epoch)
                self.model.train()
                for images, labels in self.dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.cpu().numpy()
                    labels = labels.cpu().numpy()
                    accuracy = accuracy_score(labels, predicted)
                    precision = precision_score(labels, predicted, average='weighted')
                    self.logger.info('Accuracy: %0.5f' % accuracy)
                    self.logger.info('Precision: %0.5f' % precision)
                # Evaluation on test set
                self.model.eval()
        except Exception as e:
            self.logger.error('Exception %s occurred during train_model.' % e)

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
            self.logger.info(f"{phase_name} Accuracy: {accuracy:.4f}")
            self.logger.info(f"{phase_name} Precision: {precision:.4f}")
        except Exception as e:
            self.logger.error('Exception %s occurred during check_model.' % e)


if __name__ == '__main__':
    vgg19 = VGG19()
    vgg19.load_data()
    vgg19.load_model()
    vgg19.train_model()
    vgg19.check_model(validate=True)
    # vgg19.check_model(validate=False)
