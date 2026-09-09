"""
Script para Geração Automática do Dataset de Imagens (Escalogramas CWT) do Dataset XJTU-SY.
Processa as 3 classes (normal, inner_race, outer_race),
divide ARQUIVOS .csv em conjuntos de treino, validação e teste (evitando Data Leakage),
calcula a CWT com frequências de até 10 kHz (25.6 kHz Fs) e salva imagens PNG 224x224.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pywt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image

from src.config import (
    XJTU_CLASSES,
    XJTU_PROCESSED_DIR,
    XJTU_FS,
    XJTU_FREQ_MIN,
    XJTU_FREQ_MAX,
    IMG_HEIGHT,
    IMG_WIDTH,
    WAVELET,
    RANDOM_SEED
)
from src.dataset_xjtu import load_xjtu_csv_file, segment_xjtu_signal, get_xjtu_manifest
from src.cwt_processor import scalogram_to_rgb


def get_xjtu_scales():
    """Calcula as escalas CWT para a taxa de amostragem de 25.6 kHz do XJTU-SY."""
    target_freqs = np.linspace(XJTU_FREQ_MIN, XJTU_FREQ_MAX, IMG_HEIGHT)
    scales = pywt.frequency2scale(WAVELET, target_freqs / XJTU_FS)
    return scales, target_freqs


def compute_xjtu_cwt(signal_window: np.ndarray) -> np.ndarray:
    """Calcula a magnitude da CWT para uma janela de sinal do XJTU-SY a 25.6 kHz."""
    scales, _ = get_xjtu_scales()
    coefs_cwt, _ = pywt.cwt(
        signal_window,
        scales,
        WAVELET,
        sampling_period=1.0 / XJTU_FS,
        method="conv"
    )
    return np.abs(coefs_cwt)


def create_directory_structure():
    """Cria a estrutura de pastas para o dataset XJTU-SY processado."""
    splits = ["train", "val", "test"]
    for split in splits:
        for cls_name in XJTU_CLASSES:
            path = XJTU_PROCESSED_DIR / split / cls_name
            path.mkdir(parents=True, exist_ok=True)


def process_and_generate_xjtu_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_files_per_class: int = 30
):
    """
    Gera o dataset de escalogramas CWT no domínio do tempo para o XJTU-SY sem Data Leakage.

    Args:
        train_ratio (float): Proporção de treino (padrão 0.70).
        val_ratio (float): Proporção de validação (padrão 0.15).
        test_ratio (float): Proporção de teste (padrão 0.15).
        max_files_per_class (int): Quantidade de arquivos por classe para manter balanceamento.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "As proporções devem somar 1.0"

    print("=" * 70)
    print(" GERAÇÃO DO DATASET DE ESCALOGRAMAS CWT — XJTU-SY")
    print("    Divisão estrita por ARQUIVOS .csv (Prevenção de Data Leakage)")
    print(f"    Taxa de amostragem: {XJTU_FS} Hz | Freqs: {XJTU_FREQ_MIN}-{XJTU_FREQ_MAX} Hz")
    print("=" * 70)

    create_directory_structure()
    manifest = get_xjtu_manifest(max_files_per_class=max_files_per_class)
    total_images_saved = 0

    for cls_name in XJTU_CLASSES:
        files = manifest[cls_name]
        print(f"\n[+] Processando classe '{cls_name}' ({len(files)} arquivos .csv)...")

        temp_ratio = val_ratio + test_ratio
        train_files, temp_files = train_test_split(
            files, test_size=temp_ratio, random_state=RANDOM_SEED, shuffle=True
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_ratio / temp_ratio), random_state=RANDOM_SEED, shuffle=True
        )

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, split_files in splits.items():
            dest_dir = XJTU_PROCESSED_DIR / split_name / cls_name
            img_idx = 0

            for csv_file in tqdm(split_files, desc=f"  {split_name:>5} ({cls_name})", leave=False):
                sig = load_xjtu_csv_file(csv_file, channel=0)
                windows = segment_xjtu_signal(sig)

                for w in windows:
                    cwt_mat = compute_xjtu_cwt(w)
                    img = scalogram_to_rgb(cwt_mat)
                    filename = f"{cls_name}_{split_name}_{img_idx:05d}.png"
                    img.save(str(dest_dir / filename), format="PNG")
                    img_idx += 1
                    total_images_saved += 1

            print(f"    ✅ {split_name:>5} ({cls_name}): {len(split_files)} arquivos -> {img_idx} imagens geradas")

    print("\n" + "=" * 70)
    print(f"[SUCESSO] Total de {total_images_saved} escalogramas gerados e salvos em:")
    print(f"          {XJTU_PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    process_and_generate_xjtu_dataset(max_files_per_class=30)
