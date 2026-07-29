"""
Módulo de processamento e treinamento da CNN personalizada (BearingCNN)
para classificação dos escalogramas CWT (Diagnóstico de Falhas em Rolamentos).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_CLASSES,
    DROPOUT_RATE,
    IMG_HEIGHT,
    IMG_WIDTH
)


def get_data_loaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Carrega os datasets de escalogramas CWT salvos em data/processed/ (train, val, test)
    e retorna os DataLoaders correspondentes.

    Args:
        batch_size (int): Tamanho do lote (padrão definido em config.py).

    Returns:
        Tuple contendo (train_loader, val_loader, test_loader, classes).
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"
    test_dir = PROCESSED_DATA_DIR / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Diretório de treino não encontrado em {train_dir}")

    train_dataset = ImageFolder(root=str(train_dir), transform=transform)
    val_dataset = ImageFolder(root=str(val_dir), transform=transform)
    test_dataset = ImageFolder(root=str(test_dir), transform=transform)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=use_pin_memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=use_pin_memory)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=use_pin_memory)

    return train_loader, val_loader, test_loader, train_dataset.classes


class BearingCNN(nn.Module):
    """
    Rede Convolucional para Classificação das 4 classes de falhas em rolamentos:
    - normal
    - ball
    - inner_race
    - outer_race
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
        super(BearingCNN, self).__init__()
        self.features = nn.Sequential(
            # Bloco 1: 3x224x224 -> 32x112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloco 2: 32x112x112 -> 64x56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloco 3: 64x56x56 -> 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloco 4: 128x28x28 -> 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_and_evaluate_bearing_cnn(
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE
) -> Dict[str, Any]:
    """
    Treina e avalia a arquitetura BearingCNN personalizada com checkpointing
    automático no estado ótimo (menor val_loss).

    Args:
        num_epochs (int): Número de épocas de treinamento.
        lr (float): Taxa de aprendizado do otimizador Adam.

    Returns:
        Dict contendo: model_name, history (train/val loss e acc),
        best_val_acc, test_acc, test_preds, test_labels, classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = get_data_loaders(BATCH_SIZE)
    
    model = BearingCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    checkpoint_path = Path("checkpoint_bearing_cnn_best.pth")
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    print("=" * 80)
    print(f"[INFO] EXPERIMENTO BEARING CNN (PRÓPRIA) | Épocas={num_epochs} | LR={lr}")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        # Treinamento
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0
        
        # Validação
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            status_msg = f"[CHECKPOINT] Modelo ótimo salvo na época {epoch:02d}/{num_epochs:02d}."
        else:
            status_msg = ""
            
        print(f"Época [{epoch:02d}/{num_epochs:02d}] | "
              f"Treino Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% {status_msg}")
        
    print(f"\n[INFO] Carregando pesos do melhor modelo salvo ({checkpoint_path})...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
            
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_acc = (test_preds == test_labels).mean() * 100.0
    
    print(f"[RESULTADO FINAL] Acurácia no Teste (BEARING CNN): {test_acc:.2f}%\n")
    
    return {
        "model_name": "bearing_cnn",
        "history": history,
        "best_val_acc": max(history["val_acc"]),
        "test_acc": test_acc,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "classes": classes
    }


Net = BearingCNN
