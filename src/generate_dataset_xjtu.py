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

from functools import lru_cache
import numpy as np
import pywt
from tqdm import tqdm

from src.config import (
    XJTU_CLASSES,
    XJTU_PROCESSED_DIR,
    XJTU_FS,
    XJTU_FREQ_MIN,
    XJTU_FREQ_MAX,
    IMG_HEIGHT,
    WAVELET
)
from src.dataset_xjtu import (
    load_xjtu_csv_file,
    segment_xjtu_signal,
    get_xjtu_strict_splits
)
from src.cwt_processor import scalogram_to_rgb


@lru_cache(maxsize=None)
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
    num_train_files: int = 16,
    num_val_files: int = 12,
    num_test_files: int = 10,
    max_windows_per_file: int = 15,
    max_files_per_class: int = None,
    **kwargs
):
    """
    Gera o dataset de escalogramas CWT para o XJTU-SY com Divisão Estrita Leave-Bearing-Out:
      - 9 Rolamentos Físicos Mutuamente Exclusivos (Zero vazamento de identidade).
      - Falhas Puras Comprovadas na Literatura Oficial (Wang et al., 2020 Tabela II).
      - Linha de Base Saudável Nominal Estritamente nos Arquivos Iniciais.
      - Conjunto de Teste Inédito na Condição de 40 Hz / 10 kN (2400 RPM).

    Alocação dos Rolamentos:
      - Treino: Bearing1_1 (normal), Bearing1_2 (outer_race), Bearing2_1 (inner_race)
      - Validação: Bearing2_2 (normal), Bearing2_4 (outer_race), Bearing3_3 (inner_race)
      - Teste: Bearing3_1 (normal), Bearing3_5 (outer_race), Bearing3_4 (inner_race)
    """
    print("=" * 70)
    print(" GERAÇÃO DO DATASET DE ESCALOGRAMAS CWT — XJTU-SY")
    print("    Divisão Estrita: Leave-Bearing-Out com 9 Rolamentos Disjuntos")
    print("    Falhas Puras Comprovadas (Wang et al., 2020) — Sem Falhas Combinadas")
    print(f"    Taxa de amostragem: {XJTU_FS} Hz | Freqs: {XJTU_FREQ_MIN}-{XJTU_FREQ_MAX} Hz")
    print("=" * 70)

    create_directory_structure()
    splits = get_xjtu_strict_splits(
        num_train_files=num_train_files,
        num_val_files=num_val_files,
        num_test_files=num_test_files
    )
    total_images_saved = 0

    for cls_name in XJTU_CLASSES:
        print(f"\n[+] Processando classe '{cls_name}'...")

        for split_name in ["train", "val", "test"]:
            split_files = splits[split_name][cls_name]
            dest_dir = XJTU_PROCESSED_DIR / split_name / cls_name
            img_idx = 0

            for csv_file in tqdm(split_files, desc=f"  {split_name:>5} ({cls_name})", leave=False):
                sig = load_xjtu_csv_file(csv_file, channel=0)
                windows = segment_xjtu_signal(sig)
                if max_windows_per_file is not None:
                    windows = windows[:max_windows_per_file]

                for w in windows:
                    cwt_mat = compute_xjtu_cwt(w)
                    img = scalogram_to_rgb(cwt_mat)
                    filename = f"{cls_name}_{split_name}_{img_idx:05d}.png"
                    img.save(str(dest_dir / filename), format="PNG")
                    img_idx += 1
                    total_images_saved += 1

            print(f"    [OK] {split_name:>5} ({cls_name}): {len(split_files)} arquivos -> {img_idx} imagens geradas")

    print("\n" + "=" * 70)
    print(f"[INFO] Total de {total_images_saved} escalogramas gerados e salvos em:")
    print(f"       {XJTU_PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    process_and_generate_xjtu_dataset()

