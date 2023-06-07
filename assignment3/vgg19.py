import torch
from vgg_pytorch import VGG
from data_loader import FlowerDataLoader
from sklearn.metrics import accuracy_score, precision_score

num_epochs = 1000


class VGG19:
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
            # Load the pre-trained VGG model
            self.model = VGG.from_pretrained('vgg19', num_classes=102)  # Update num_classes to match your dataset
        except Exception as e:
            print('Exception %s occurred during load_model.' % e)

    def train_model(self):
        try:
            # Define the loss function and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            # Training loop
            for epoch in range(num_epochs):
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
                    print('Accuracy:', accuracy)
                    print('Precision:', precision)
                # Evaluation on test set
                self.model.eval()
        except Exception as e:
            print('Exception %s occurred during train_model.' % e)


if __name__ == '__main__':
    vgg19 = VGG19()
    vgg19.load_data()
    vgg19.load_model()
    vgg19.train_model()
