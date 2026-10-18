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

from functools import lru_cache
import numpy as np
import pywt
from tqdm import tqdm

from src.config import (
    PADERBORN_CLASSES,
    PADERBORN_PROCESSED_DIR,
    PADERBORN_FS,
    PADERBORN_FREQ_MIN,
    PADERBORN_FREQ_MAX,
    IMG_HEIGHT,
    WAVELET
)
from src.dataset_paderborn import (
    load_paderborn_mat_file,
    segment_paderborn_signal,
    get_paderborn_strict_splits
)
from src.cwt_processor import scalogram_to_rgb


@lru_cache(maxsize=None)
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
        method="fft"
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
    num_train_files: int = 8,
    num_val_files: int = 4,
    num_test_files: int = 5,
    max_windows_per_file: int = 25,
    max_files_per_class: int = None,
    **kwargs
):
    """
    Gera o dataset de escalogramas CWT para o Paderborn com Divisão Estrita Inter-Condições Operacionais:
      - Treino: Condição Nominal N15_M07_F10 (1500 RPM, 0.7 Nm, 1000 N).
      - Validação: Condição N15_M07_F04 (1500 RPM, 0.7 Nm, Força Radial Reduzida 400 N).
      - Teste: Condição Inédita N15_M01_F10 (1500 RPM, Torque Reduzido 0.1 Nm, 1000 N).

    Avalia com rigor a generalização do diagnóstico sob variação de torque de carga e força radial.

    Args:
        num_train_files (int): Quantidade de arquivos de treino por classe (padrão: 8).
        num_val_files (int): Quantidade de arquivos de validação por classe (padrão: 4).
        num_test_files (int): Quantidade de arquivos de teste por classe (padrão: 5).
        max_windows_per_file (int, optional): Limite de janelas por arquivo para balanceamento (padrão: 25).
    """
    print("=" * 70)
    print(" GERAÇÃO DO DATASET DE ESCALOGRAMAS CWT — PADERBORN (PU DATASET)")
    print("    Divisão Estrita: Leave-Operating-Condition-Out (1500 RPM)")
    print("    Treino: N15_M07_F10 | Validação: N15_M07_F04 | Teste: N15_M01_F10")
    print(f"    Taxa de amostragem: {PADERBORN_FS} Hz | Freqs: {PADERBORN_FREQ_MIN}-{PADERBORN_FREQ_MAX} Hz")
    print("=" * 70)

    create_directory_structure()
    splits = get_paderborn_strict_splits(
        num_train_files=num_train_files,
        num_val_files=num_val_files,
        num_test_files=num_test_files
    )

    total_images_saved = 0

    for cls_name in PADERBORN_CLASSES:
        print(f"\n[+] Processando classe: '{cls_name}'...")

        for split_name in ["train", "val", "test"]:
            file_list = splits[split_name][cls_name]
            file_names = [f.stem[:18] for f in file_list]
            print(f"    {split_name:>5}: {len(file_list)} arquivos -> {file_names}")

            dest_dir = PADERBORN_PROCESSED_DIR / split_name / cls_name
            idx = 0

            for mat_file in file_list:
                signal = load_paderborn_mat_file(mat_file)
                windows = segment_paderborn_signal(signal)
                if max_windows_per_file is not None:
                    windows = windows[:max_windows_per_file]

                for window in tqdm(windows, desc=f"      {cls_name}/{split_name}/{mat_file.stem[:18]}", leave=False):
                    scalogram_mat = compute_paderborn_cwt(window)
                    img = scalogram_to_rgb(scalogram_mat)
                    
                    img_path = dest_dir / f"{cls_name}_{split_name}_{idx:05d}.png"
                    img.save(str(img_path), format="PNG")
                    idx += 1
                    total_images_saved += 1

            print(f"    [OK] {split_name:>5}: {len(file_list)} arquivos -> {idx} imagens geradas")

    print("\n" + "=" * 70)
    print(f" Concluído: Total de {total_images_saved} imagens salvas em:")
    print(f" {PADERBORN_PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    process_and_generate_paderborn_dataset()

