"""
Script para Geração Automática do Dataset de Imagens (Escalogramas) para Treinamento da CNN.
Processa todas as classes do CWRU, divide ARQUIVOS .mat em conjuntos de treino, validação
e teste (evitando Data Leakage), e salva as imagens PNG (224x224) em pastas organizadas.
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
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED
)
from src.dataset import load_cwru_mat_file, segment_signal
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
    Pipeline de geração do dataset SEM Data Leakage:
    1. Para cada classe, lista os arquivos .mat disponíveis.
    2. Divide os ARQUIVOS .mat entre train/val/test (não as janelas!).
    3. Somente depois aplica o janelamento deslizante em cada split.
    4. Calcula a CWT e salva os escalogramas PNG 224x224.

    Isso garante que nenhuma amostra do sinal de treino apareça no teste,
    mesmo com 50% de sobreposição (overlap) no janelamento.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "As proporções devem somar 1.0"

    print("==========================================================")
    print(" GERAÇÃO DO DATASET DE ESCALOGRAMAS (CWT -> CNN) ")
    print("    Divisão por ARQUIVOS (sem Data Leakage) ")
    print("==========================================================")

    create_directory_structure()

    total_images_saved = 0

    for cls_name in CLASSES:
        print(f"\n[+] Processando classe: '{cls_name}'...")

        # 1. Listar todos os arquivos .mat da classe
        class_dir = RAW_DATA_DIR / cls_name
        mat_files = sorted(class_dir.glob("*.mat"))
        print(f"    Arquivos .mat encontrados: {len(mat_files)}")

        # 2. Dividir ARQUIVOS (não janelas!) entre train / val / test
        temp_ratio = val_ratio + test_ratio
        train_files, temp_files = train_test_split(
            mat_files,
            test_size=temp_ratio,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        relative_val_ratio = val_ratio / temp_ratio
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1.0 - relative_val_ratio),
            random_state=RANDOM_SEED,
            shuffle=True
        )

        file_splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        # Log dos arquivos por split
        for split_name, file_list in file_splits.items():
            file_names = [f.stem for f in file_list]
            print(f"    {split_name:>5}: {len(file_list)} arquivos → {file_names}")

        # 3. Para cada split: carregar arquivos → janelamento → CWT → salvar PNG
        for split_name, file_list in file_splits.items():
            dest_dir = PROCESSED_DATA_DIR / split_name / cls_name
            idx = 0

            for mat_file in file_list:
                # Carregar sinal bruto e aplicar janelamento
                signal = load_cwru_mat_file(mat_file)
                windows = segment_signal(signal)

                for window in tqdm(windows, desc=f"      {cls_name}/{split_name}/{mat_file.stem}"):
                    # Calcular CWT
                    scalogram_mat = compute_cwt(window)

                    # Salvar PNG RGB 224x224 sem bordas
                    img_path = dest_dir / f"{cls_name}_{split_name}_{idx:04d}.png"
                    save_scalogram_png(scalogram_mat, str(img_path))
                    idx += 1
                    total_images_saved += 1

            print(f"    ✅ {split_name}: {idx} imagens geradas")

    print("\n==========================================================")
    print(f" Processo Concluído! Total de {total_images_saved} imagens salvas em:")
    print(f" {PROCESSED_DATA_DIR}")
    print("==========================================================")


if __name__ == "__main__":
    process_and_generate_dataset()

