# Importing necessary libraries
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

# Function to create a dataframe from image files in a folder
def create_dataframe_from_folder(folder):
    data = []
    # Iterate through each label (folder name)
    for label in os.listdir(folder):
        class_folder = os.path.join(folder, label)
        # Check if it's a directory
        if os.path.isdir(class_folder):
            # Iterate through each image file in the folder
            for filename in os.listdir(class_folder):
                # Check for specific file extensions
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(class_folder, filename)
                    data.append((image_path, label))

    # Convert the data to a pandas dataframe and shuffle it
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    return shuffle(df)

# Function to preprocess images for input into a neural network
def preprocess_images(df, target_size=(224, 224)):
    # Define a series of transformations for the images
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images, labels = [], []
    # Iterate through the dataframe and apply transformations
    for image_path, label in df.values:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
        labels.append(label)

    # Encode labels using one-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_labels = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    return torch.stack(images), torch.tensor(encoded_labels)

# Training and validation functions

# Function to train the model
def train_model(train_images, train_labels, val_images, val_labels, num_epochs, batch_size, num_classes):
    # Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load a pretrained ResNet152 model
    model = models.resnet152(pretrained=True)
    # Replace the fully connected layer for our specific task
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare data loaders for training and validation
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Move model to the specified device
    model.to(device)
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Iterate over the training data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels_max).sum().item()

        # Calculate loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch: {epoch + 1}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            # Iterate over the validation data
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                # Accumulate accuracy metrics
                _, predicted = torch.max(outputs.data, 1)
                _, labels_max = torch.max(labels.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels_max).sum().item()

        # Calculate loss and accuracy for the validation data
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = 100 * val_correct / val_total
        print(f"Epoch: {epoch + 1}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%")
        # Set model back to train mode
        model.train()

    return model

# Function to test the model
def test_model(test_images, test_labels, batch_size, num_classes):
    # Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load a pretrained ResNet152 model
    model = models.resnet152(pretrained=True)
    # Replace the fully connected layer for our specific task
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Load the trained model
    model.load_state_dict(torch.load(model_path))  
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # Prepare the test data loader
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        # Iterate over the test data
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()

            # Accumulate accuracy metrics
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels_max).sum().item()

        # Calculate loss and accuracy for the test data
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_predictions / total_predictions
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Argument Parsing

# Initialize an argument parser for command line argument processing
parser = argparse.ArgumentParser()
# Define command line arguments and their default values
parser.add_argument('--epochs', type=int, default=10)  # Number of training epochs
parser.add_argument('--batch-size', type=int, default=32)  # Batch size for training
parser.add_argument('--num-classes', type=int, default=3)  # Number of classes in the dataset
args = parser.parse_args()  # Parse the arguments from the command line

# Set data directories

# Retrieve training, validation, and test directories from environment variables
train_dir = os.environ['SM_CHANNEL_TRAIN']
valid_dir = os.environ['SM_CHANNEL_VALID']
test_dir = os.environ['SM_CHANNEL_TEST']

# Load and preprocess data

# Create dataframes from images in training and validation directories
df_train = create_dataframe_from_folder(train_dir)
df_validation = create_dataframe_from_folder(valid_dir)

# Select a subset (20%) of training and validation data for faster processing
df_train = df_train.sample(frac=0.2)  # 20% of training data
df_validation = df_validation.sample(frac=0.2)  # 20% of validation data

# Preprocess the training and validation images
train_images, train_labels = preprocess_images(df_train)
val_images, val_labels = preprocess_images(df_validation)

# Load and preprocess test data

# Create a dataframe from images in the test directory
df_test = create_dataframe_from_folder(test_dir)

# Select a subset (40%) of test data
df_test = df_test.sample(frac=0.4)  # 40% of test data

# Preprocess the test images
test_images, test_labels = preprocess_images(df_test)

# Train the model with validation data
model = train_model(train_images, train_labels, val_images, val_labels, args.epochs, args.batch_size, args.num_classes)

# Save the model

# Set the default model directory in SageMaker
model_dir = '/opt/ml/model'  
# Define the full path for saving the model
model_path = os.path.join(model_dir, 'model.pth')
# Save the model's state dictionary
torch.save(model.state_dict(), model_path)

# Test the model
test_model(test_images, test_labels, args.batch_size, args.num_classes)