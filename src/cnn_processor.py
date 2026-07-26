"""
Módulo de processamento e treinamento da CNN (Rede Neural Convolucional)
para classificação dos escalogramas CWT (Diagnóstico de Falhas em Rolamentos).
"""

import sys
from pathlib import Path

# Garantir importação do módulo src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Importar todas as configurações centralizadas de src/config.py
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


def get_data_loaders(batch_size: int = BATCH_SIZE):
    """
    Carrega os datasets de escalogramas CWT salvos em data/processed/ (train, val, test)
    e retorna os DataLoaders correspondentes.
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

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Alias de compatibilidade
Net = BearingCNN


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")

    train_loader, val_loader, test_loader, classes = get_data_loaders(BATCH_SIZE)
    print(f"Classes identificadas: {classes}")
    print(f"Lotes de treino: {len(train_loader)}, Validação: {len(val_loader)}, Teste: {len(test_loader)}")

    model = BearingCNN(num_classes=len(classes)).to(device)
    print("\nEstrutura do Modelo:")
    print(model)
