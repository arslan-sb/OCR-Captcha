import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os

# Define your custom dataset
class OCRDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images and labels
        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(image_folder, filename)
                label = os.path.splitext(filename)[0]  # Use the image name as the label
                self.images.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define your model (example: a simple CNN)
class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Adjust to the number of classes

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = x.view(-1, 32 * 50 * 50)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Set up training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100, 100))
])

dataset = OCRDataset('/home/arslan/DIP Projects/FINAL PROJECT/MobileNet Fine-tuning/train/65L', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Count the number of unique labels
unique_labels = set(dataset.labels)
num_classes = len(unique_labels)
print(f"Number of unique labels: {num_classes}")

# Encode labels as integers
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Initialize model with the correct number of classes
model = OCRModel(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):  # Number of epochs
    epoch_loss = 0.0
    for images, labels in dataloader:
        # Convert labels to indices
        labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Print actual and predicted labels
        _, predicted_indices = torch.max(outputs, 1)
        predicted_labels = [index_to_label[idx.item()] for idx in predicted_indices]
        actual_labels = [index_to_label[idx.item()] for idx in labels]

        for actual, predicted in zip(actual_labels, predicted_labels):
            print(f'Actual: {actual}, Predicted: {predicted}')

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

# Save the model
torch.save(model.state_dict(), 'ocr_model.pth')
