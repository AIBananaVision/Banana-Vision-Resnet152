

class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = transforms.ToTensor()(tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        # return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std), std={1})'.format(self.mean, self.std)
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

import os
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import argparse
from sagemaker.s3 import S3Uploader

# Helper functions

def create_dataframe_from_folder(folder):
    data = []
    for label in os.listdir(folder):
        class_folder = os.path.join(folder, label)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(class_folder, filename)
                    data.append((image_path, label))

    df = pd.DataFrame(data, columns=['image_path', 'label'])
    return shuffle(df)


def preprocess_images(df, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        GaussianNoiseTransform(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ... [Other transformations if any]
    ])
    # ... [Rest of the preprocessing code]

    # ])

    images, labels = [], []
    for image_path, label in df.values:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
        labels.append(label)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_labels = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    return torch.stack(images), torch.tensor(encoded_labels)

# Training and validation functions

# Training and validation functions

def train_model(train_images, train_labels, val_images, val_labels, num_epochs, batch_size, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels_max).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch: {epoch + 1}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, labels_max = torch.max(labels.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels_max).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = 100 * val_correct / val_total
        print(f"Epoch: {epoch + 1}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%")
        model.train()

    return model



def test_model(test_images, test_labels, batch_size, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))  # Load the trained model
    # model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels_max).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_predictions / total_predictions
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    
# Argument Parsing

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-classes', type=int, default=3)
args = parser.parse_args()

# Set data directories

train_dir = os.environ['SM_CHANNEL_TRAIN']
valid_dir = os.environ['SM_CHANNEL_VALID']
test_dir = os.environ['SM_CHANNEL_TEST']

# Load and preprocess data

df_train = create_dataframe_from_folder(train_dir)
df_validation = create_dataframe_from_folder(valid_dir)


# Select only 20% of training data
# df_train = df_train.sample(frac=0.2)

# Select only 20% of validation data
# df_validation = df_validation.sample(frac=0.2)

train_images, train_labels = preprocess_images(df_train)
val_images, val_labels = preprocess_images(df_validation)

# Load and preprocess test data
df_test = create_dataframe_from_folder(test_dir)

# Select only 20% of test data
# df_test = df_test.sample(frac=0.4)


test_images, test_labels = preprocess_images(df_test)

# Train the model with validation data
model = train_model(train_images, train_labels, val_images, val_labels, args.epochs, args.batch_size, args.num_classes)

# Test the model
# test_model(test_images, test_labels, args.batch_size, args.num_classes)

# Save the model

model_dir = '/opt/ml/model'  # Default model directory in SageMaker
model_path = os.path.join(model_dir, 'model.pth')
torch.save(model.state_dict(), model_path)


# Test the model
test_model(test_images, test_labels, args.batch_size, args.num_classes)
