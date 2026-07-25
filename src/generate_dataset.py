"""
Script para Geração Automática do Dataset de Imagens (Escalogramas) para Treinamento da CNN.
Processa todas as classes do CWRU, divide em conjuntos de treino, validação e teste,
e salva as imagens PNG (224x224) em pastas organizadas.
"""

import sys
from pathlib import Path

# Adicionar a raiz do projeto ao PYTHONPATH para importar src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.config import (
    CLASSES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED
)
from src.dataset import load_class_windows
from src.cwt_processor import compute_cwt, save_scalogram_png


def create_directory_structure():
    """Cria as pastas de destino para o dataset processado (train, val, test)."""
    splits = ["train", "val", "test"]
    for split in splits:
        for cls_name in CLASSES:
            path = PROCESSED_DATA_DIR / split / cls_name
            path.mkdir(parents=True, exist_ok=True)


def process_and_generate_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Carrega todas as janelas das 4 classes, calcula a CWT de cada uma,
    divide entre train/val/test e salva os escalogramas PNG 224x224.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "As proporções devem somar 1.0"
    
    print("==========================================================")
    print(" GERAÇÃO DO DATASET DE ESCALOGRAMAS (CWT -> CNN) ")
    print("==========================================================")
    
    create_directory_structure()
    
    total_images_saved = 0
    
    for cls_name in CLASSES:
        print(f"\n[+] Processando classe: '{cls_name}'...")
        windows = load_class_windows(cls_name)
        print(f"    Total de janelas extraídas: {len(windows)}")
        
        # Divisão Train / Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_wins, temp_wins = train_test_split(
            windows,
            test_size=temp_ratio,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        
        # Divisão Val / Test
        relative_val_ratio = val_ratio / temp_ratio
        val_wins, test_wins = train_test_split(
            temp_wins,
            test_size=(1.0 - relative_val_ratio),
            random_state=RANDOM_SEED,
            shuffle=True
        )
        
        splits = {
            "train": train_wins,
            "val": val_wins,
            "test": test_wins
        }
        
        for split_name, win_list in splits.items():
            print(f"    Generating {split_name} ({len(win_list)} imagens)...")
            dest_dir = PROCESSED_DATA_DIR / split_name / cls_name
            
            for idx, window in enumerate(tqdm(win_list, desc=f"      {cls_name}/{split_name}")):
                # Calcular CWT
                scalogram_mat = compute_cwt(window)
                
                # Salvar PNG RGB 224x224 sem bordas
                img_path = dest_dir / f"{cls_name}_{split_name}_{idx:04d}.png"
                save_scalogram_png(scalogram_mat, str(img_path))
                total_images_saved += 1
                
    print("\n==========================================================")
    print(f" Processo Concluído! Total de {total_images_saved} imagens salvas em:")
    print(f" {PROCESSED_DATA_DIR}")
    print("==========================================================")


if __name__ == "__main__":
    process_and_generate_dataset()
