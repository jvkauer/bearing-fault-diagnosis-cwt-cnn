"""
Módulo Unificado de Treinamento e Avaliação — Trainer Engine.
Consolida o loop de treinamento com checkpointing automático (val_loss),
eliminando duplicação entre cnn_processor.py e transfer_learning.py.

Uso futuro: substituir as funções train_and_evaluate_* dos módulos individuais
por chamadas a este trainer unificado.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Callable

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
    NUM_EPOCHS,
    LEARNING_RATE,
)


# ==============================================================================
# 1. DataLoader Factory
# ==============================================================================

def create_dataloaders(
    img_size: int = 224,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Cria DataLoaders para os conjuntos train/val/test a partir de data/processed/.

    Args:
        img_size (int): Dimensão espacial de entrada da rede (224 ou 299).
        batch_size (int): Tamanho do lote.
        num_workers (int): Número de workers paralelos para carregamento de dados.

    Returns:
        Tuple contendo (train_loader, val_loader, test_loader, classes).

    Raises:
        FileNotFoundError: Se os diretórios train/val/test não existirem.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"
    test_dir = PROCESSED_DATA_DIR / "test"

    for split_dir, split_name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Diretório de {split_name} não encontrado em {split_dir}. "
                f"Execute generate_dataset.py primeiro."
            )

    train_ds = ImageFolder(root=str(train_dir), transform=transform)
    val_ds = ImageFolder(root=str(val_dir), transform=transform)
    test_ds = ImageFolder(root=str(test_dir), transform=transform)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin_memory)

    return train_loader, val_loader, test_loader, train_ds.classes


# ==============================================================================
# 2. Training Engine
# ==============================================================================

def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    classes: List[str],
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    checkpoint_path: Optional[Path] = None,
    optimizer_factory: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Loop unificado de treinamento, validação e teste com checkpointing automático
    baseado na menor val_loss.

    Args:
        model (nn.Module): Modelo PyTorch já instanciado e movido para o device.
        model_name (str): Nome do modelo para logging e identificação.
        train_loader (DataLoader): DataLoader do conjunto de treinamento.
        val_loader (DataLoader): DataLoader do conjunto de validação.
        test_loader (DataLoader): DataLoader do conjunto de teste.
        classes (List[str]): Lista de nomes das classes.
        num_epochs (int): Número de épocas de treinamento.
        lr (float): Taxa de aprendizado do otimizador Adam.
        checkpoint_path (Path, optional): Caminho para salvar o checkpoint.
            Se None, usa 'checkpoint_{model_name}_best.pth'.
        optimizer_factory (Callable, optional): Função que recebe os parâmetros
            do modelo e retorna um otimizador. Se None, usa Adam com lr.

    Returns:
        Dict contendo:
            - model_name (str)
            - history (dict): Listas de train_loss, val_loss, train_acc, val_acc.
            - best_val_acc (float)
            - test_acc (float)
            - test_preds (np.ndarray)
            - test_labels (np.ndarray)
            - classes (List[str])
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    if checkpoint_path is None:
        checkpoint_path = Path(f"checkpoint_{model_name}_best.pth")

    if optimizer_factory is not None:
        optimizer = optimizer_factory(model.parameters())
    else:
        # Filtra apenas parâmetros que requerem gradiente (para transfer learning)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=lr)

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print("=" * 80)
    print(f"[INFO] EXPERIMENTO: {model_name.upper()} | Épocas={num_epochs} | LR={lr}")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        # ------------------------------------------------------------------
        # Fase de Treinamento
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Fase de Validação
        # ------------------------------------------------------------------
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

        # Checkpointing (menor val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            status_msg = f"[CHECKPOINT] Modelo ótimo salvo na época {epoch:02d}/{num_epochs:02d}."
        else:
            status_msg = ""

        print(f"Época [{epoch:02d}/{num_epochs:02d}] | "
              f"Treino Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% {status_msg}")

    # ------------------------------------------------------------------
    # Fase de Teste (com pesos do melhor checkpoint)
    # ------------------------------------------------------------------
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
