import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ================================
# 1. Кастомный датасет для иерархической структуры
# ================================
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Рекурсивно сканирует root_dir и собирает все изображения.
        Имя последней папки используется как метка.
        """
        self.samples = []  # список кортежей: (путь к изображению, label_index)
        self.class_to_idx = {}
        self.idx_to_class = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Если в текущей папке есть изображения, считаем её классом
            if filenames:
                label = os.path.basename(dirpath)
                if label not in self.class_to_idx:
                    idx = len(self.class_to_idx)
                    self.class_to_idx[label] = idx
                    self.idx_to_class[idx] = label
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.samples.append((os.path.join(dirpath, file), self.class_to_idx[label]))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Ошибка открытия {path}: {e}")
            # В случае ошибки возвращаем пустое изображение фиксированного размера
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, label

# ================================
# 2. Функции для обучения и валидации
# ================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, idx_to_class):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    class_report = classification_report(all_labels, all_preds, target_names=target_names)
    return epoch_loss, epoch_acc, epoch_f1, conf_matrix, class_report

# ================================
# 3. Главная функция обучения
# ================================
def main():
    # Гиперпараметры
    dataset_root = r"NewDataset_balanced"
    input_size = 224
    batch_size = 32
    num_epochs = 25
    learning_rate = 1e-4
    weight_decay = 1e-4

    # Устройство (выводится один раз)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Трансформации для обучения и валидации
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = AnimalDataset(dataset_root, transform=train_transform)
    num_classes = len(full_dataset.class_to_idx)
    print(f"Найдено {num_classes} классов: {full_dataset.class_to_idx}")

    idx_to_class = full_dataset.idx_to_class

    # Разбиваем датасет на обучающую (80%) и валидационную (20%) выборки
    train_size_val = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size_val
    train_dataset, val_dataset = random_split(full_dataset, [train_size_val, val_size])
    # Для валидации используем другой набор трансформаций:
    val_dataset.dataset.transform = val_transform

    # DataLoader-ы
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, conf_matrix, class_report = validate_epoch(model, val_loader, criterion, device, idx_to_class)
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%")
        print(f"  Val loss:   {val_loss:.4f}, Val acc:   {val_acc:.2f}%")
        print(f"  Val F1-score (macro): {val_f1:.4f}")
        print("  Confusion Matrix:")
        print(conf_matrix)
        print("  Classification Report:")
        print(class_report)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_efficientnet.pth")
            print("  Модель сохранена.")

    print("Обучение завершено.")


if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
