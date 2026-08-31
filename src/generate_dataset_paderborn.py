"""
Script para Geração Automática do Dataset de Imagens (Escalogramas CWT) do Dataset de Paderborn.
Processa as classes (K001 -> normal, KI14 -> inner_race, KA15 -> outer_race),
divide ARQUIVOS .mat em conjuntos de treino, validação e teste (evitando Data Leakage),
calcula a CWT com frequências de até 20 kHz (64 kHz Fs) e salva imagens PNG 224x224.
"""

import sys
from pathlib import Path

# Adicionar a raiz do projeto ao PYTHONPATH para importar src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pywt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

from src.config import (
    PADERBORN_CLASSES,
    PADERBORN_FOLDER_MAP,
    PADERBORN_RAW_DIR,
    PADERBORN_PROCESSED_DIR,
    PADERBORN_FS,
    PADERBORN_FREQ_MIN,
    PADERBORN_FREQ_MAX,
    IMG_HEIGHT,
    IMG_WIDTH,
    WAVELET,
    RANDOM_SEED
)
from src.dataset_paderborn import load_paderborn_mat_file, segment_paderborn_signal
from src.cwt_processor import scalogram_to_rgb


def get_paderborn_scales():
    """Calcula as escalas CWT para a taxa de amostragem de 64 kHz do Paderborn."""
    target_freqs = np.linspace(PADERBORN_FREQ_MIN, PADERBORN_FREQ_MAX, IMG_HEIGHT)
    scales = pywt.frequency2scale(WAVELET, target_freqs / PADERBORN_FS)
    return scales, target_freqs


def compute_paderborn_cwt(signal_window: np.ndarray) -> np.ndarray:
    """Calcula a CWT para uma janela de sinal do Paderborn a 64 kHz."""
    scales, _ = get_paderborn_scales()
    coefs_cwt, _ = pywt.cwt(
        signal_window,
        scales,
        WAVELET,
        sampling_period=1.0 / PADERBORN_FS,
        method="conv"
    )
    return np.abs(coefs_cwt)


def create_directory_structure():
    """Cria a estrutura de pastas para o dataset Paderborn processado."""
    splits = ["train", "val", "test"]
    for split in splits:
        for cls_name in PADERBORN_CLASSES:
            path = PADERBORN_PROCESSED_DIR / split / cls_name
            path.mkdir(parents=True, exist_ok=True)


def process_and_generate_paderborn_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_files_per_class: int = None
):
    """
    Gera o dataset de escalogramas CWT para o Paderborn sem Data Leakage.

    Args:
        train_ratio (float): Proporção de treino (padrão 0.70).
        val_ratio (float): Proporção de validação (padrão 0.15).
        test_ratio (float): Proporção de teste (padrão 0.15).
        max_files_per_class (int, optional): Limite de arquivos por classe para testes rápidos.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "As proporções devem somar 1.0"

    print("=" * 70)
    print(" GERAÇÃO DO DATASET DE ESCALOGRAMAS CWT — PADERBORN (PU DATASET)")
    print("    Divisão por ARQUIVOS .mat (Prevenção de Data Leakage)")
    print(f"    Taxa de amostragem: {PADERBORN_FS} Hz | Freqs: {PADERBORN_FREQ_MIN}-{PADERBORN_FREQ_MAX} Hz")
    print("=" * 70)

    create_directory_structure()
    total_images_saved = 0

    for folder_name, cls_name in PADERBORN_FOLDER_MAP.items():
        print(f"\n[+] Processando pasta '{folder_name}' -> Classe: '{cls_name}'...")
        class_raw_dir = PADERBORN_RAW_DIR / folder_name
        
        if not class_raw_dir.exists():
            print(f"    [AVISO] Pasta {class_raw_dir} não encontrada! Pulando.")
            continue

        mat_files = sorted(class_raw_dir.glob("*.mat"))
        if max_files_per_class:
            mat_files = mat_files[:max_files_per_class]

        print(f"    Total de arquivos .mat selecionados: {len(mat_files)}")

        # 1. Divisão por arquivos (SEM Data Leakage)
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

        for split_name, file_list in file_splits.items():
            dest_dir = PADERBORN_PROCESSED_DIR / split_name / cls_name
            idx = 0

            for mat_file in file_list:
                signal = load_paderborn_mat_file(mat_file)
                windows = segment_paderborn_signal(signal)

                for window in tqdm(windows, desc=f"      {cls_name}/{split_name}/{mat_file.stem[:18]}", leave=False):
                    scalogram_mat = compute_paderborn_cwt(window)
                    img = scalogram_to_rgb(scalogram_mat)
                    
                    img_path = dest_dir / f"{cls_name}_{split_name}_{idx:05d}.png"
                    img.save(str(img_path), format="PNG")
                    idx += 1
                    total_images_saved += 1

            print(f"    ✅ {split_name:>5}: {len(file_list)} arquivos -> {idx} imagens geradas")

    print("\n" + "=" * 70)
    print(f" Processamento Concluído! Total de {total_images_saved} imagens salvas em:")
    print(f" {PADERBORN_PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    process_and_generate_paderborn_dataset()
