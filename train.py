import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
from torchvision import models, transforms

def load_images_with_labels(folder):
    images = []
    labels = []
    for label in ['fire', 'nofire']:
        path = os.path.join(folder, label)
        print(f"Loading images from: {path}")  # Diagnostyka
        files = glob(os.path.join(path, '*.jpg'))
        print(f"Found {len(files)} images in {path}")  # Diagnostyka
        for filename in files:
            img = cv2.imread(filename)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(label)
            else:
                print(f"Failed to load image: {filename}")  # Diagnostyka
    return np.array(images), np.array(labels)

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    data_dir = "/app/data/forest_fire"
    train_dir = os.path.join(data_dir, 'Training and Validation')
    test_dir = os.path.join(data_dir, 'Testing')

    all_images, all_labels = load_images_with_labels(train_dir)
    print(f"Loaded {len(all_images)} training images")  # Diagnostyka

    test_images, test_labels = load_images_with_labels(test_dir)
    print(f"Loaded {len(test_images)} testing images")  # Diagnostyka

    if len(all_images) == 0 or len(test_images) == 0:
        raise ValueError("No images found in the specified directories. Please check the directory paths and contents.")

    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels_encoded, test_size=0.2, stratify=all_labels_encoded, random_state=42
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(train_images, train_labels, transform=transform)
    val_dataset = CustomDataset(val_images, val_labels, transform=transform)
    test_dataset = CustomDataset(test_images, test_labels_encoded, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0001)

    device = torch.device("cpu")
    model.to(device)

    num_epochs = 10
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_accuracy = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
              f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

    print("Training complete")

    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))

    y_true = np.array(all_labels)
    y_pred_proba = np.array(all_probs)

    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    print(f'ROC-AUC: {roc_auc}')

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

if __name__ == "__main__":
    main()
