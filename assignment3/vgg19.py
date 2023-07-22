import os.path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from vgg_pytorch import VGG

from assignment3 import data_downloader
from logger_utils import LoggerUtils
from sklearn.metrics import accuracy_score


class VGG19:
    def __init__(self, train_folder: str):
        self.dataset_folder = train_folder

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_data_loader = DataLoader(ImageFolder(os.path.join(self.dataset_folder, 'train'),
                                                        transform=transform),
                                            batch_size=64, shuffle=True)
        self.val_data_loader = DataLoader(ImageFolder(os.path.join(self.dataset_folder, 'val'),
                                                      transform=transform),
                                          batch_size=64, shuffle=True)
        self.test_data_loader = DataLoader(ImageFolder(os.path.join(self.dataset_folder, 'test'),
                                                       transform=transform),
                                           batch_size=64, shuffle=True)
        self.model = None
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='vgg19.log')

    def load_model(self):
        try:
            # Load the pre-trained VGG model
            self.model = VGG.from_pretrained('vgg19', num_classes=102)  # Update num_classes to match your dataset
        except Exception as e:
            self.logger.error('Exception %s occurred during load_model.' % e)

    def train_model(self, num_epochs=5):
        metrics = []
        val_metrics = []
        test_metrics = []
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Define the loss function and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            accuracy = running_loss = 0
            top_acc = 0
            ckpt_last, ckpt_best = 'last.pt', 'best.pt'
            self.model.to(device)
            # Training loop
            for epoch in range(1, num_epochs + 1):
                self.logger.info('Running on epoch %d.' % epoch)
                self.model.train()
                batch_idx = 0
                for images, labels in self.train_data_loader:
                    self.logger.info('Running on batch %d' % batch_idx)
                    batch_idx += 1
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.cpu().numpy()
                    labels = labels.cpu().numpy()
                    accuracy += accuracy_score(labels, predicted)
                    running_loss += loss.item()

                accuracy /= batch_idx
                running_loss /= batch_idx

                self.logger.info('Accuracy: %0.5f' % accuracy)
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")
                metrics.append((running_loss, accuracy))
                vloss, vacc = self.check_model(validate=True)
                val_metrics.append((vloss, vacc))
                if top_acc < vacc:
                    top_acc = vacc
                tloss, tacc = self.check_model(validate=False)
                test_metrics.append((tloss, tacc))

                torch.save(self.model, ckpt_last)
                if top_acc == vacc:
                    torch.save(self.model, ckpt_best)

        except Exception as e:
            self.logger.error('Exception %s occurred during train_model.' % e)

        with open('metrics.csv', 'w') as f:
            f.write("Epoch,loss,accuracy,val_loss,val_accuracy,test_loss,test_accuracy\n")
            for epoch, ((train_loss, train_acc), (val_loss, val_acc), (test_loss, test_acc)) \
                    in enumerate(zip(metrics, val_metrics, test_metrics)):
                f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{test_loss:.4f},{test_acc:.4f}\n")

    def check_model(self, validate=True):
        try:
            # Get the validation data loader
            if validate:
                phase_name = 'Validation'
                data_loader = self.val_data_loader
            else:
                phase_name = 'Test'
                data_loader = self.test_data_loader
            # Set the model to evaluation mode
            self.model.eval()
            # Lists to store the predicted labels and true labels
            predicted_labels = []
            true_labels = []
            criterion = torch.nn.CrossEntropyLoss()
            loss = 0
            # Iterate over the validation data
            for images, labels in data_loader:
                # Move the data to the GPU, if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass to obtain the predictions
                with torch.no_grad():
                    outputs = self.model(images)
                    loss += criterion(outputs, labels).item()
                # Get the predicted labels
                _, predicted = torch.max(outputs.data, 1)
                # Append the predicted and true labels to the respective lists
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            loss /= len(data_loader)
            # Calculate the validation accuracy and precision
            accuracy = accuracy_score(true_labels, predicted_labels)
            # Print results
            self.logger.info(f"{phase_name} Loss: {loss:.4f}")
            self.logger.info(f"{phase_name} Accuracy: {accuracy:.4f}")
            return loss, accuracy
        except Exception as e:
            self.logger.error('Exception %s occurred during check_model.' % e)


if __name__ == '__main__':
    # data downloading
    data_downloader.download_images()
    labels = data_downloader.get_labels()

    train_set, val_set, test_set = data_downloader.random_split(list(range(1, labels.shape[0] + 1)),
                                                                train_sz=0.5, val_sz=0.25)

    data_downloader.format_folder('jpg', 'datasets', train_set, val_set, test_set, labels)

    vgg19 = VGG19("datasets")
    vgg19.load_model()
    vgg19.train_model(num_epochs=20)

    best_model = torch.load('best.pt')
    loss, accuracy = best_model.check_model(validate=False)
    print(f"Test {loss=}, {accuracy=}")