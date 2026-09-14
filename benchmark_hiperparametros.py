import sys
import time
from pathlib import Path

# Garantir que a raiz do projeto esteja no sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from src.config import RANDOM_SEED, set_seed
from src.transfer_learning import get_transfer_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print("=" * 80)
print(f" BENCHMARK DE HIPERPARÂMETROS — RESNET-18 (CWRU)")
print(f" Dispositivo: {device} ({gpu_name}) | Seed Fixa: {RANDOM_SEED}")
print("=" * 80)


def evaluate_config(name, lr, dropout, weight_decay, epochs):
    set_seed(RANDOM_SEED)
    start_time = time.time()
    
    train_loader, val_loader, test_loader, classes = get_transfer_dataloaders(img_size=224, seed=RANDOM_SEED)
    
    # Construir ResNet-18 com o Dropout desejado
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, len(classes)))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = None
    
    print(f"\n[+] Rodando: {name}")
    print(f"    LR={lr} | Dropout={dropout} | Weight Decay={weight_decay} | Épocas={epochs}")
    
    for epoch in range(1, epochs + 1):
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
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Avaliar no teste
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()
    
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
            
    test_acc = (test_correct / test_total) * 100.0
    elapsed = time.time() - start_time
    print(f"    --> Acurácia Teste: {test_acc:.2f}% | Val Acc: {best_val_acc:.2f}% | Tempo: {elapsed:.1f}s")
    
    return {
        "name": name,
        "lr": lr,
        "dropout": dropout,
        "wd": weight_decay,
        "epochs": epochs,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "time": elapsed
    }


if __name__ == "__main__":
    configs = [
        ("1. Baseline Atual", 1e-4, 0.5, 0.0, 10),
        ("2. Dropout 0.3", 1e-4, 0.3, 0.0, 10),
        ("3. Dropout 0.2", 1e-4, 0.2, 0.0, 10),
        ("4. LR 5e-5 + Dropout 0.3", 5e-5, 0.3, 0.0, 10),
        ("5. Dropout 0.3 + Weight Decay 1e-4", 1e-4, 0.3, 1e-4, 10),
        ("6. 15 Épocas + Dropout 0.3", 1e-4, 0.3, 0.0, 15),
    ]
    
    results = []
    for name, lr, drop, wd, ep in configs:
        res = evaluate_config(name, lr, drop, wd, ep)
        results.append(res)
        
    print("\n" + "=" * 80)
    print(" TABELA COMPARATIVA DE HIPERPARÂMETROS (RESNET-18)")
    print("=" * 80)
    print(f"{'Configuração':<35} | {'Val Acc (%)':<12} | {'Teste Acc (%)':<14} | {'Tempo'}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<35} | {r['val_acc']:>10.2f}% | {r['test_acc']:>12.2f}% | {r['time']:>5.1f}s")
    print("=" * 80)
