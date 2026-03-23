import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np

# -------------------------
# CONFIGURAÇÃO
# -------------------------
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# TRANSFORMS
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -------------------------
# DATASETS
# -------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

num_classes = len(train_dataset.classes)

print("Classes:", train_dataset.classes)

# -------------------------
# CALCULAR PESOS DAS CLASSES
# -------------------------
targets = [label for _, label in train_dataset.samples]
class_counts = np.bincount(targets)

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

# -------------------------
# MODELO
# -------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model = model.to(DEVICE)

# -------------------------
# LOSS + OTIMIZADOR
# -------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# TREINAMENTO
# -------------------------
best_acc = 0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader):

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------
    # VALIDAÇÃO
    # -------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for imgs, labels in val_loader:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # salvar melhor modelo
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_retinopathy_model.pth")
        print("✅ Melhor modelo salvo!")

# -------------------------
# TESTE FINAL
# -------------------------
print("\nTestando modelo...")

model.load_state_dict(torch.load("best_retinopathy_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total

print(f"\nTest Accuracy: {test_acc:.4f}")
