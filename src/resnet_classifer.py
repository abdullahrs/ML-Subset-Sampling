from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from torch import no_grad as torch_no_grad
from torch import save as torch_save
from torch import load as torch_load
from torch import max as torch_max
from torch import device, cuda

from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

import matplotlib.pyplot as plt
from time import time

import torch


class ImageClassifier:
    def __init__(self, batch_size: int, num_epochs: int,
                 learning_rate: float, dataset_paths: dict[str, str],
                 start_num_samples=100, sample_multiplier=2, normal_model_epoch=10,
                 load_train_model=False, load_subset_model=False, 
                 path_to_train_model=None, path_to_normal_model=None) -> None:
        
        self.batch_size = batch_size
        # Number of epochs is how many times training data will be iterated per subset  
        self.num_epochs = num_epochs
        # Number of epochs is how many times data will be iterated for normal model
        self.normal_model_epoch = normal_model_epoch
        # Learning rate is the rate at which the model will be updated
        self.learning_rate = learning_rate
        # Start num samples is the initial length of the subset
        self.start_num_samples = start_num_samples
        # Sample multiplier is how much the subset will be increased by each iteration 
        # (Iteration used to increase the subset length term wise it refers the while loop)
        self.sample_multiplier = sample_multiplier
        # load_train_model is bool value if true then the model
        # will be loaded from the path_to_train_model
        # else the model will be trained from scratch
        # Its same for load_subset_model & path_to_normal_model
        """
        !!! If there is a missing epoch in the loading section of the saved model, 
            it does not continue from there. After loading model it proceeds to testing part.
            And it just impelemented for train_model_with_subset_sampling even other params created.
        """
        self.load_train_model = load_train_model
        self.path_to_normal_model = path_to_train_model
        self.load_subset_model = load_subset_model
        self.path_to_subset_model = path_to_normal_model
        """
        Dataset paths are the paths of the train, val, test dataset
        (Due to ImageNet datase is huge we use just train_path,
        we are spliting train_data set into train, validation and test)
        dataset_paths = {
            'train_path' : str,
        }
        """
        self.train_path = dataset_paths['train_path']
        # Get the device info
        self.device = device("cuda" if cuda.is_available() else "cpu")
        print(f'Device: {self.device}')
        # Criterion for the loss function
        self.criterion = CrossEntropyLoss()
        # Lose and accuricies for the subset model
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.normal_train_losses, self.normal_val_losses, self.normal_test_losses = [], [], []
        # Lose and accuricies for the normal model
        self.train_accuracies, self.val_accuracies, self.test_accuracies = [], [], []
        self.normal_train_accuracies, self.normal_val_accuracies, self.normal_test_accuracies = [], [], []

    # Loading the data and preprocessing the data
    def load_data_and_preprocessing(self):
        # Transforms used for preprocessing the data (Resize, Crop, ToTensor, Normalize)
        # Resize the images to 256x256 and crop the center 224x224 for resnet18 model
        # resnet18 model accepts 224x224 images 
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        # Gets the root directory of the dataset
        """
        root/
            class_1/
            class_2/
            class_3/
            .../
            class_n/
        """
        self.dataset = ImageFolder(root=self.train_path, transform=transform)
        # Calculate the length of the subset and multiply by some constant
        # to reduce the number of samples in the dataset (initial dataset has over 1.2m data
        # It takes forever the train on PC)
        # %25 data = 112400 / %23 train data = 103408 , Validation Data: 14772 , Test Data: 29546 ==> Total: 149408
        # 1024 4048 16192 64768 259072
        # 5 times 16 epoch for subset sampling total 80
        # 80 times for normal model
        # ---------------------------------------------------------------------------------------------------------
        # %15 train data = 67440 , Validation Data: 9634 , Test Data: 19269 ==> Total: 149408
        # 1024 4048 16192 64768 259072
        # 4 times 16 epoch for subset sampling total 64
        # 64 times for normal model
        subset_len = int(len(self.dataset) * 0.15)

        # Create a random subset
        subset_dataset = random_split(
            self.dataset, [subset_len, len(self.dataset) - subset_len])
        # We get the first part of the dataset with length subset_len
        resized_dataset = subset_dataset[0]
        # Deciding train, validation and test set length
        train_len = int(len(subset_dataset[0]) * 0.7)  # %70 train
        val_len = int(len(subset_dataset[0]) * 0.1)  # %10 val
        test_len = len(subset_dataset[0]) - train_len - val_len  # ~%20 test
        # Splitting the dataset into train, validation and test
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            resized_dataset, [train_len, val_len, test_len])

        print(f"Train data set length: {len(self.train_dataset)}")
        print(f"Validation data set length: {len(self.val_dataset)}")
        print(f"Test data set length: {len(self.test_dataset)}")
        # Create the dataloader for train, validation and test data
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True)
        print(" Train, Validation and Test Data loaded ".center(100, '='))
        self.class_labels = self.dataset.classes
        print(
            f" Number of classes: {len(self.class_labels)} ".center(100, '='))
        # Creating the initial model
        self.model = resnet18()
        # Change the model to the device
        self.model.to(self.device)
        # Optimizer for the model
        self.optimizer = SGD(self.model.parameters(),
                             lr=self.learning_rate, momentum=0.9)

    def train_model_with_subset_sampling(self):
        print(f" Model with Subset Sampling Started ".center(100, '='))
        train_loss, val_loss, test_loss = [], [], []
        subset_index = self.start_num_samples
        subset_indices = list(range(subset_index))
        current_subset = Subset(self.train_dataset, subset_indices)
        number_of_train_element = len(self.train_dataset)

        while (len(current_subset) < number_of_train_element and not self.load_subset_model):
            print(
                f"Length of current subset is {len(current_subset)}, train data set length is {number_of_train_element}".center(100, '-'))
            self.train_loader = DataLoader(
                current_subset, batch_size=64, shuffle=True)

            for epoch in range(self.num_epochs):
                start_time = time()
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for images, labels in self.train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    _, predicted = torch_max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_loss += loss.item() * images.size(0)

                train_loss /= len(self.train_loader.dataset)
                train_accuracy = train_correct / train_total

                val_loss, val_accuracy = self.evaluate(self.val_loader)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_accuracy)
                self.val_accuracies.append(val_accuracy)
                self.save_model(
                    f'data/train_model_with_subset_sampling_resnet-18-{epoch}.pth')
                end_time = time()
                print(f"Subset Size: {len(current_subset)} | "
                      f"Epoch {epoch+1}/{self.num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
                      f" | Epoch completed in: {end_time-start_time:.2f}s")

            subset_index *= self.sample_multiplier
            subset_index = subset_index if subset_index < number_of_train_element else number_of_train_element
            subset_indices = list(range(subset_index))
            current_subset = Subset(self.train_dataset, subset_indices)
        if self.load_subset_model:
            self.load_model(
                'data/train_model_with_subset_sampling_resnet-18-15.pth')
        test_loss, test_accuracy = self.evaluate(self.test_loader)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)
        self.save_model(
            'data/train_model_with_subset_sampling-final.pth')

        print(
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
        return self.model

    def evaluate(self, data_loader):
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0

        with torch_no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * images.size(0)

                _, predicted = torch_max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss /= len(data_loader.dataset)
        accuracy = correct / total

        return loss, accuracy

    def train_model(self):
        print(f" Train model started ".center(100, '='))
        self.model = resnet18()
        self.model.to(self.device)
        self.optimizer = SGD(self.model.parameters(),
                             lr=self.learning_rate, momentum=0.9)

        for epoch in range(self.normal_model_epoch):
            start_time = time()
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch_max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item() * images.size(0)

            train_loss /= len(self.train_loader.dataset)
            train_accuracy = train_correct / train_total

            val_loss, val_accuracy = self.evaluate(self.val_loader)
            self.normal_train_losses.append(train_loss)
            self.normal_val_losses.append(val_loss)
            self.normal_train_accuracies.append(train_accuracy)
            self.normal_val_accuracies.append(val_accuracy)
            self.save_model(f'data/train_model_resnet-18-{epoch}.pth')
            end_time = time()
            print(f"Epoch {epoch+1}/{self.normal_model_epoch} | "
                  f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
                  f" | Epoch completion time: {end_time - start_time:.2f}s")

        test_loss, test_accuracy = self.evaluate(self.test_loader)
        self.normal_test_losses.append(test_loss)
        self.normal_test_accuracies.append(test_accuracy)
        self.save_model('data/train_model_resnet-18-epochs.pth')
        print(
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
        return train_loss, val_loss, test_loss

    def save_model(self, path):
        # save_path = 'custom-classifier_resnet_18_epochs.pth'
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch_load(path))
        print(f"Model loaded from {path}")

    def list_average(self, l, r):
        length = len(l)
        if length // r == 0:
            return l
        div = length // r
        result = []
        for i in range(r):
            result.append(0)
            result[i] = sum([l[i+j*div] for j in range(div)])
        return result

    def show_results(self):
        average_train_losses = self.list_average(
            self.train_losses, self.num_epochs)
        average_val_losses = self.list_average(
            self.val_losses, self.num_epochs)
        average_train_accuracies = self.list_average(
            self.train_accuracies, self.num_epochs)
        average_val_accuracies = self.list_average(
            self.val_accuracies, self.num_epochs)

        with open('results.txt', 'a') as file:
            file.write(f'train_losses: {self.train_losses}')
            file.write('\n')
            file.write(f'train_accuracies: {self.train_accuracies}')
            file.write('\n')
            file.write(f'normal_train_losses: {self.normal_train_losses}')
            file.write('\n')
            file.write(
                f'normal_train_accuracies: {self.normal_train_accuracies}')
            file.write('\n')
            file.write(f'val_losses: {self.val_losses}')
            file.write('\n')
            file.write(f'val_accuracies: {self.val_accuracies}')
            file.write('\n')
            file.write(f'normal_val_loses: {self.normal_val_losses}')
            file.write('\n')
            file.write(f'normal_val_accuracies: {self.normal_val_accuracies}')
            file.write('\n')
            file.write(f'test_losses: {self.test_losses}')
            file.write('\n')
            file.write(f'test_accuracies: {self.test_accuracies}')
            file.write('\n')
            file.write(f'normal_test_losses: {self.normal_test_losses}')
            file.write('\n')
            file.write(
                f'normal_test_accuracies: {self.normal_test_accuracies}')
            file.write('\n')
            file.write(f'average_train_losses: {average_train_losses}')
            file.write('\n')
            file.write(f'average_val_losses: {average_val_losses}')
            file.write('\n')
            file.write(f'average_train_accuracies: {average_train_accuracies}')
            file.write('\n')
            file.write(f'average_val_accuracies: {average_val_accuracies}')
            file.write('\n')
            file.write('\n')
            file.write('::'*64)

        self.plot_metric(range(1, self.num_epochs + 1),
                         self.train_losses, 'Loses', 'Train Loses')
        self.plot_metric(range(1, self.num_epochs + 1),
                         self.val_losses, 'Loses', 'Validation Loses')
        self.plot_metric(range(1, self.num_epochs + 1),
                         self.train_accuracies, 'Accuracies', 'Train Accuracies')
        self.plot_metric(range(1, self.num_epochs + 1),
                         self.val_accuracies, 'Accuracies', 'Validation Accuracies')
        self.plot_metric(range(1, self.normal_model_epoch + 1),
                         self.normal_train_losses, 'Loses', 'Normal Training Loses')
        self.plot_metric(range(1, self.normal_model_epoch + 1),
                         self.normal_val_losses, 'Loses', 'Normal Validation Loses')
        self.plot_metric(range(1, self.normal_model_epoch + 1),
                         self.normal_train_accuracies, 'Accuracies', 'Normal Training Accuracies')
        self.plot_metric(range(1, self.normal_model_epoch + 1),
                         self.normal_val_accuracies, 'Accuracies', 'Normal Validation Accuracies')

        self.compare_results(epochs=range(1, self.num_epochs + 1), result1=self.train_losses, result2=self.val_losses, result1_label='Train Loses',
                             result2_label='Validation Loses',
                             label='Subset Sampling Losses')

        self.compare_results(range(1, self.num_epochs + 1), self.train_accuracies,
                             self.val_accuracies,
                             'Train Accuracies',
                             'Validation Accuracies',
                             'Subset Sampling Accuracies')

        self.compare_results(range(1, self.normal_model_epoch + 1), self.normal_train_losses,
                             self.normal_val_losses,
                             'Training Loses',
                             'Validation Loses',
                             'Normal Training Loses')

        self.compare_results(range(1, self.normal_model_epoch + 1), self.normal_train_accuracies,
                             self.normal_val_accuracies,
                             'Training Accuracies',
                             'Validation Accuracies',
                             'Normal Training Accuracies')

        self.compare_results(range(1, self.num_epochs + 1), self.train_losses, self.train_accuracies,
                             'Train Loses',
                             'Train Accuracies',
                             'Subset Sampling Training Losses and Accuracies')

        self.compare_results(range(1, self.num_epochs + 1), self.val_losses, self.val_accuracies,
                             'Validation Loses',
                             'Validation Accuracies',
                             'Subset Sampling Validation Losses and Accuracies')

    def plot_metric(self, epochs, values, data_label: str, plot_label: str):
        plt.plot(epochs, values)
        plt.title(plot_label)
        plt.xlabel("Epochs")
        plt.ylabel(data_label)
        plt.show()

    def compare_results(self, epochs, result1, result2, result1_label: str, result2_label: str, label: str):
        plt.plot(epochs, result1, label=result1_label)
        plt.plot(epochs, result2, label=result2_label)

        plt.title(label)
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def run_test(self):
        times = []
        start_time = time()
        self.load_data_and_preprocessing()
        end_time = time()
        times.append(end_time - start_time)
        start_time = time()
        self.train_model_with_subset_sampling()
        end_time = time()
        times.append(end_time - start_time)
        start_time = time()
        self.train_model()
        end_time = time()
        times.append(end_time - start_time)
        print(
            f"Total load and preprocessing time: {times[0]} ".ljust(100, '='))
        print(
            f"Total train_model_with_subset_sampling time: {times[1]} ".ljust(100, '='))
        print(f"Total train_model time: {times[2]} ".ljust(100, '='))
        self.show_results()
