"""
Módulo de Transfer Learning para Diagnóstico de Falhas em Rolamentos (CWRU).
Implementa construtores e o pipeline de treinamento com checkpointing automático.
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
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES
)


def get_transfer_dataloaders(img_size: int = 224, batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Retorna DataLoaders com o tamanho de imagem específico do modelo (224x224 ou 299x299).

    Args:
        img_size (int): Dimensão espacial de entrada da rede (224 ou 299).
        batch_size (int): Tamanho do lote.

    Returns:
        Tuple contendo (train_loader, val_loader, test_loader, classes).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"
    test_dir = PROCESSED_DATA_DIR / "test"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Diretório de treino não encontrado em {train_dir}")
        
    train_ds = ImageFolder(root=str(train_dir), transform=transform)
    val_ds = ImageFolder(root=str(val_dir), transform=transform)
    test_ds = ImageFolder(root=str(test_dir), transform=transform)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=use_pin_memory)
    
    return train_loader, val_loader, test_loader, train_ds.classes


def build_transfer_model(model_name: str, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> Tuple[nn.Module, int]:
    """
    Constrói e adapta modelos pré-treinados no ImageNet para a classificação do CWRU.

    Args:
        model_name (str): Nome do modelo ('resnet18', 'inception_v3', 'efficientnet_b0').
        num_classes (int): Número de classes de saída.
        freeze_backbone (bool): Se True, congela os pesos do backbone pré-treinado.

    Returns:
        Tuple contendo (model, img_size) — modelo adaptado e tamanho de entrada.
    """
    model_name = model_name.lower()
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
        img_size = 224
        
    elif model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        model.AuxLogits = None
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
        img_size = 299
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
        img_size = 224
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")
        
    return model, img_size


def train_and_evaluate_transfer_model(
    model_name: str,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    freeze: bool = True
) -> Dict[str, Any]:
    """
    Treina e avalia modelos de Transfer Learning com checkpointing automático
    no estado ótimo (menor val_loss).

    Args:
        model_name (str): Nome do modelo ('resnet18', 'inception_v3', 'efficientnet_b0').
        num_epochs (int): Número de épocas de treinamento.
        lr (float): Taxa de aprendizado do otimizador Adam.
        freeze (bool): Se True, congela o backbone (feature extraction).

    Returns:
        Dict contendo: model_name, history (train/val loss e acc),
        best_val_acc, test_acc, test_preds, test_labels, classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = build_transfer_model(model_name, num_classes=4, freeze_backbone=freeze)
    model = model.to(device)
    
    train_loader, val_loader, test_loader, classes = get_transfer_dataloaders(img_size=img_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    best_val_loss = float("inf")
    checkpoint_path = Path(f"checkpoint_{model_name}_best.pth")
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    print("=" * 80)
    print(f"[INFO] EXPERIMENTO TRANSFER LEARNING: {model_name.upper()} | Freeze={freeze} | Épocas={num_epochs} | LR={lr}")
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
    
    print(f"[RESULTADO FINAL] Acurácia no Teste ({model_name.upper()}): {test_acc:.2f}%\n")
    
    return {
        "model_name": model_name,
        "history": history,
        "best_val_acc": max(history["val_acc"]),
        "test_acc": test_acc,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "classes": classes
    }
