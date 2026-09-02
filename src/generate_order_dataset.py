"""
Script de Geração de Datasets de Escalogramas Ângulo-Ordem (Order-CWT) para CWRU e Paderborn.
Aplica Computed Order Tracking (COT) com reamostragem angular invariante à rotação (RPM),
garantindo divisão estrita por arquivos .mat para evitar Data Leakage.

v2 (Otimizado): Normalização GLOBAL (two-pass amostrado) + Z-score para preservar amplitude cross-domain.
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
    elif file_name.startswith("N12"):
        return 1200.0
    elif file_name.startswith("N15"):
        return 1500.0
    return 900.0


# =============================================================================
#  Funções auxiliares para o Two-Pass com Z-Score Global
# =============================================================================

def _compute_cwru_global_signal_stats():
    """Calcula média e desvio padrão GLOBAIS sobre todos os sinais do CWRU."""
    print("[Z-SCORE] Calculando estatísticas globais do CWRU...")
    all_vals = []
    for cls_name in CWRU_CLASSES:
        class_dir = CWRU_DIR / cls_name
        for mat_file in sorted(class_dir.glob("*.mat")):
            sig = load_cwru_mat_file(mat_file)
            all_vals.append(sig)
    all_vals = np.concatenate(all_vals)
    mean, std = float(np.mean(all_vals)), float(np.std(all_vals))
    print(f"    CWRU global: mean={mean:.6f}, std={std:.6f} ({len(all_vals)} amostras)")
    return mean, std


def _compute_paderborn_global_signal_stats(max_files_per_class=10):
    """Calcula média e desvio padrão GLOBAIS sobre os sinais do Paderborn."""
    print("[Z-SCORE] Calculando estatísticas globais do Paderborn...")
    all_vals = []
    for folder_name in PADERBORN_FOLDER_MAP.keys():
        class_dir = PADERBORN_RAW_DIR / folder_name
        mat_files = sorted(class_dir.glob("*.mat"))
        if max_files_per_class:
            mat_files = mat_files[:max_files_per_class]
        for mat_file in mat_files:
            sig = load_paderborn_mat_file(mat_file)
            all_vals.append(sig)
    all_vals = np.concatenate(all_vals)
    mean, std = float(np.mean(all_vals)), float(np.std(all_vals))
    print(f"    Paderborn global: mean={mean:.6f}, std={std:.6f} ({len(all_vals)} amostras)")
    return mean, std


def _cwru_order_cwt_pass1(global_mean, global_std, samples_per_file=4):
    """Calcula vmin/vmax globais amostrando chunks representativos do CWRU."""
    print("[PASS 1] Estimando vmin/vmax globais do CWRU Order-CWT...")
    all_p1, all_p99 = [], []

    for cls_name in CWRU_CLASSES:
        class_dir = CWRU_DIR / cls_name
        for mat_file in sorted(class_dir.glob("*.mat")):
            rpm = extract_cwru_rpm(mat_file.name)
            sig = load_cwru_mat_file(mat_file)
            sig = (sig - global_mean) / global_std

            fr = rpm / 60.0
            samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * CWRU_FS))
            max_starts = len(sig) - samples_per_window_time
            if max_starts <= 0:
                continue

            starts = np.linspace(0, max_starts, samples_per_file, dtype=int)
            for start in starts:
                chunk = sig[start: start + samples_per_window_time]
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
                all_p1.append(np.percentile(cwt_mat, 1))
                all_p99.append(np.percentile(cwt_mat, 99))

    global_vmin = float(np.percentile(all_p1, 5))
    global_vmax = float(np.percentile(all_p99, 95))
    print(f"    CWRU Order-CWT estimado: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
    return global_vmin, global_vmax


def _paderborn_order_cwt_pass1(global_mean, global_std, max_files_per_class=10, samples_per_file=4):
    """Calcula vmin/vmax globais amostrando chunks representativos do Paderborn."""
    print("[PASS 1] Estimando vmin/vmax globais do Paderborn Order-CWT...")
    all_p1, all_p99 = [], []

    for folder_name in PADERBORN_FOLDER_MAP.keys():
        class_dir = PADERBORN_RAW_DIR / folder_name
        mat_files = sorted(class_dir.glob("*.mat"))
        if max_files_per_class:
            mat_files = mat_files[:max_files_per_class]

        for mat_file in mat_files:
            rpm = extract_paderborn_rpm(mat_file.name)
            sig = load_paderborn_mat_file(mat_file)
            sig = (sig - global_mean) / global_std

            fr = rpm / 60.0
            samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * PADERBORN_FS))
            max_starts = len(sig) - samples_per_window_time
            if max_starts <= 0:
                continue

            starts = np.linspace(0, max_starts, samples_per_file, dtype=int)
            for start in starts:
                chunk = sig[start: start + samples_per_window_time]
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
                all_p1.append(np.percentile(cwt_mat, 1))
                all_p99.append(np.percentile(cwt_mat, 99))

    global_vmin = float(np.percentile(all_p1, 5))
    global_vmax = float(np.percentile(all_p99, 95))
    print(f"    Paderborn Order-CWT estimado: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
    return global_vmin, global_vmax


# =============================================================================
#  PASSO 2: Geração com Normalização Global
# =============================================================================

def generate_cwru_order_dataset(
    global_mean: float,
    global_std: float,
    global_vmin: float,
    global_vmax: float,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """Gera escalogramas de ordens para o dataset CWRU com normalização GLOBAL."""
    print("=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — CWRU [GLOBAL NORM]")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
    print(f"    Z-Score: mean={global_mean:.4f}, std={global_std:.4f}")
    print(f"    Colormap Norm: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
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

                # Z-score global
                sig = (sig - global_mean) / global_std

                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * CWRU_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start: min(start + samples_per_window_time + 8, len(sig))]
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
                    # Normalização GLOBAL
                    img = scalogram_to_rgb(cwt_mat, vmin=global_vmin, vmax=global_vmax)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    ✅ CWRU {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens CWRU geradas em: {ORDER_CWRU_DIR} ({total_images} imagens)")


def generate_paderborn_order_dataset(
    global_mean: float,
    global_std: float,
    global_vmin: float,
    global_vmax: float,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_files_per_class: int = 10
):
    """Gera escalogramas de ordens para o dataset Paderborn com normalização GLOBAL."""
    print("\n" + "=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — PADERBORN [GLOBAL NORM]")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
    print(f"    Z-Score: mean={global_mean:.4f}, std={global_std:.4f}")
    print(f"    Colormap Norm: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
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

                # Z-score global
                sig = (sig - global_mean) / global_std

                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * PADERBORN_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start: min(start + samples_per_window_time + 8, len(sig))]
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
                    # Normalização GLOBAL
                    img = scalogram_to_rgb(cwt_mat, vmin=global_vmin, vmax=global_vmax)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    ✅ Paderborn {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

def generate_all_order_datasets(max_files_per_class: int = 10):
    """
    Pipeline completo de geração dos datasets Order-CWT para CWRU e Paderborn:
    1. Z-Score Global dos sinais 1D
    2. Estimativa Amostrada de vmin/vmax globais compartilhados
    3. Geração e gravação dos escalogramas PNG com normalização global
    """
    cwru_mean, cwru_std = _compute_cwru_global_signal_stats()
    pad_mean, pad_std = _compute_paderborn_global_signal_stats(max_files_per_class=max_files_per_class)

    cwru_vmin, cwru_vmax = _cwru_order_cwt_pass1(cwru_mean, cwru_std, samples_per_file=4)
    pad_vmin, pad_vmax = _paderborn_order_cwt_pass1(pad_mean, pad_std, max_files_per_class=max_files_per_class, samples_per_file=4)

    shared_vmin = min(cwru_vmin, pad_vmin)
    shared_vmax = max(cwru_vmax, pad_vmax)
    print(f"\n[GLOBAL COMPARTILHADO] vmin={shared_vmin:.6f}, vmax={shared_vmax:.6f}\n")

    generate_cwru_order_dataset(cwru_mean, cwru_std, shared_vmin, shared_vmax)
    generate_paderborn_order_dataset(pad_mean, pad_std, shared_vmin, shared_vmax, max_files_per_class=max_files_per_class)


if __name__ == "__main__":
    generate_all_order_datasets(max_files_per_class=10)
