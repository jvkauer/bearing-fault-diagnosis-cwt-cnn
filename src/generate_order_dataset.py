"""
Script de Geração de Datasets de Escalogramas Ângulo-Ordem (Order-CWT) para CWRU e Paderborn.
Aplica Computed Order Tracking (COT) com reamostragem angular invariante à rotação (RPM),
garantindo divisão estrita por arquivos .mat para evitar Data Leakage.
"""

import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image

from src.config import (
    CWRU_DIR,
    CLASSES as CWRU_CLASSES,
    PADERBORN_CLASSES,
    PADERBORN_FOLDER_MAP,
    PADERBORN_RAW_DIR,
    ORDER_CWRU_DIR,
    ORDER_PADERBORN_DIR,
    FS as CWRU_FS,
    PADERBORN_FS,
    ORDER_SAMPLES_PER_REV,
    ORDER_NUM_REVS,
    ORDER_MIN,
    ORDER_MAX,
    RANDOM_SEED
)
from src.dataset_cwru import load_cwru_mat_file
from src.dataset_paderborn import load_paderborn_mat_file
from src.order_tracking import resample_to_angle_domain, compute_order_cwt
from src.cwt_processor import scalogram_to_rgb


def extract_cwru_rpm(file_name: str) -> float:
    """Extrai o RPM do nome do arquivo CWRU (ex: 'B007_0HP_1797rpm.mat' -> 1797.0)."""
    match = re.search(r"(\d+)rpm", file_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    if "0HP" in file_name:
        return 1797.0
    elif "1HP" in file_name:
        return 1772.0
    elif "2HP" in file_name:
        return 1750.0
    elif "3HP" in file_name:
        return 1730.0
    return 1772.0


def extract_paderborn_rpm(file_name: str) -> float:
    """Extrai o RPM do nome do arquivo Paderborn (ex: 'N09_M07_F10_K001_1.mat' -> 900.0)."""
    if file_name.startswith("N09"):
        return 900.0
    elif file_name.startswith("N15"):
        return 1500.0
    return 900.0


def generate_cwru_order_dataset(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """Gera escalogramas de ordens para o dataset CWRU."""
    print("=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — CWRU")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        for cls_name in CWRU_CLASSES:
            (ORDER_CWRU_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

    total_images = 0

    for cls_name in CWRU_CLASSES:
        class_dir = CWRU_DIR / cls_name
        mat_files = sorted(class_dir.glob("*.mat"))
        if not mat_files:
            continue

        temp_ratio = val_ratio + test_ratio
        train_files, temp_files = train_test_split(
            mat_files, test_size=temp_ratio, random_state=RANDOM_SEED, shuffle=True
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_ratio / temp_ratio), random_state=RANDOM_SEED, shuffle=True
        )

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in splits.items():
            dest = ORDER_CWRU_DIR / split_name / cls_name
            idx = 0

            for mat_file in files:
                rpm = extract_cwru_rpm(mat_file.name)
                sig = load_cwru_mat_file(mat_file)
                
                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * CWRU_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start : start + samples_per_window_time]
                    ang_sig = resample_to_angle_domain(
                        chunk, CWRU_FS, rpm,
                        samples_per_rev=ORDER_SAMPLES_PER_REV,
                        num_revs=ORDER_NUM_REVS
                    )
                    cwt_mat, _ = compute_order_cwt(
                        ang_sig,
                        samples_per_rev=ORDER_SAMPLES_PER_REV,
                        order_min=ORDER_MIN,
                        order_max=ORDER_MAX
                    )
                    img = scalogram_to_rgb(cwt_mat)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    ✅ CWRU {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens CWRU geradas em: {ORDER_CWRU_DIR} ({total_images} imagens)")


def generate_paderborn_order_dataset(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_files_per_class: int = 10
):
    """Gera escalogramas de ordens para o dataset Paderborn."""
    print("\n" + "=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — PADERBORN")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        for cls_name in PADERBORN_CLASSES:
            (ORDER_PADERBORN_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

    total_images = 0

    for folder_name, cls_name in PADERBORN_FOLDER_MAP.items():
        class_dir = PADERBORN_RAW_DIR / folder_name
        mat_files = sorted(class_dir.glob("*.mat"))
        if max_files_per_class:
            mat_files = mat_files[:max_files_per_class]

        temp_ratio = val_ratio + test_ratio
        train_files, temp_files = train_test_split(
            mat_files, test_size=temp_ratio, random_state=RANDOM_SEED, shuffle=True
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_ratio / temp_ratio), random_state=RANDOM_SEED, shuffle=True
        )

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in splits.items():
            dest = ORDER_PADERBORN_DIR / split_name / cls_name
            idx = 0

            for mat_file in files:
                rpm = extract_paderborn_rpm(mat_file.name)
                sig = load_paderborn_mat_file(mat_file)

                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * PADERBORN_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start : start + samples_per_window_time]
                    ang_sig = resample_to_angle_domain(
                        chunk, PADERBORN_FS, rpm,
                        samples_per_rev=ORDER_SAMPLES_PER_REV,
                        num_revs=ORDER_NUM_REVS
                    )
                    cwt_mat, _ = compute_order_cwt(
                        ang_sig,
                        samples_per_rev=ORDER_SAMPLES_PER_REV,
                        order_min=ORDER_MIN,
                        order_max=ORDER_MAX
                    )
                    img = scalogram_to_rgb(cwt_mat)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    ✅ Paderborn {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens Paderborn geradas em: {ORDER_PADERBORN_DIR} ({total_images} imagens)")


if __name__ == "__main__":
    generate_cwru_order_dataset()
    generate_paderborn_order_dataset(max_files_per_class=10)
