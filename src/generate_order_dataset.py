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

from src.config import (
    CWRU_DIR,
    CLASSES as CWRU_CLASSES,
    PADERBORN_CLASSES,
    PADERBORN_FOLDER_MAP,
    PADERBORN_RAW_DIR,
    ORDER_CWRU_DIR,
    ORDER_PADERBORN_DIR,
    ORDER_XJTU_DIR,
    XJTU_CLASSES,
    FS as CWRU_FS,
    PADERBORN_FS,
    XJTU_FS,
    ORDER_SAMPLES_PER_REV,
    ORDER_NUM_REVS,
    ORDER_MIN,
    ORDER_MAX,
    RANDOM_SEED
)
from src.dataset_cwru import load_cwru_mat_file, get_cwru_strict_splits
from src.dataset_paderborn import load_paderborn_mat_file, get_paderborn_strict_splits
from src.dataset_xjtu import load_xjtu_csv_file, extract_xjtu_rpm, get_xjtu_manifest, get_xjtu_strict_splits
from src.order_tracking import resample_to_angle_domain, compute_order_cwt
from src.cwt_processor import scalogram_to_rgb


_CWRU_NUMERIC_RPM_MAP = {
    "97": 1797.0, "105": 1797.0, "118": 1797.0, "130": 1797.0,
    "169": 1797.0, "185": 1797.0, "197": 1797.0, "209": 1797.0, "222": 1797.0, "234": 1797.0,
    "98": 1772.0, "106": 1772.0, "119": 1772.0, "131": 1772.0,
    "170": 1772.0, "186": 1772.0, "198": 1772.0, "210": 1772.0, "223": 1772.0, "235": 1772.0,
    "99": 1750.0, "107": 1750.0, "120": 1750.0, "132": 1750.0,
    "171": 1750.0, "187": 1750.0, "199": 1750.0, "211": 1750.0, "224": 1750.0, "236": 1750.0,
    "100": 1730.0, "108": 1730.0, "121": 1730.0, "133": 1730.0,
    "172": 1730.0, "188": 1730.0, "200": 1730.0, "212": 1730.0, "225": 1730.0, "237": 1730.0,
}


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
    stem = Path(file_name).stem
    if stem in _CWRU_NUMERIC_RPM_MAP:
        return _CWRU_NUMERIC_RPM_MAP[stem]
    raise ValueError(f"Não foi possível identificar o RPM no nome do arquivo CWRU: {file_name}")


def extract_paderborn_rpm(file_name: str) -> float:
    """Extrai o RPM do nome do arquivo Paderborn (ex: 'N09_M07_F10_K001_1.mat' -> 900.0)."""
    if file_name.startswith("N09"):
        return 900.0
    elif file_name.startswith("N12"):
        return 1200.0
    elif file_name.startswith("N15"):
        return 1500.0
    elif file_name.startswith("N04"):
        return 400.0
    raise ValueError(f"Não foi possível identificar o RPM no nome do arquivo Paderborn: {file_name}")



# =============================================================================
#  Funções auxiliares para o Two-Pass com Z-Score Global Livre de Data Leakage
# =============================================================================

def _split_files_list(
    files: list,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> dict:
    """Divide lista de arquivos em train, val e test evitando data leakage."""
    temp_ratio = val_ratio + test_ratio
    train_files, temp_files = train_test_split(
        files, test_size=temp_ratio, random_state=seed, shuffle=True
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=(test_ratio / temp_ratio), random_state=seed, shuffle=True
    )
    return {"train": train_files, "val": val_files, "test": test_files}


def _compute_cwru_global_signal_stats(train_files_by_class: dict = None):
    """Calcula média e desvio padrão GLOBAIS sobre os sinais do CWRU (estritamente partição de treino)."""
    print("[Z-SCORE] Calculando estatísticas globais do CWRU (apenas partição de treino)...")
    if train_files_by_class is None:
        train_files_by_class = get_cwru_strict_splits()["train"]

    all_vals = []
    for cls_name in CWRU_CLASSES:
        mat_files = train_files_by_class.get(cls_name, [])
        for mat_file in mat_files:
            sig = load_cwru_mat_file(mat_file)
            all_vals.append(sig)
    all_vals = np.concatenate(all_vals)
    mean, std = float(np.mean(all_vals)), float(np.std(all_vals))
    print(f"    CWRU global (treino): mean={mean:.6f}, std={std:.6f} ({len(all_vals)} amostras)")
    return mean, std


def _compute_paderborn_global_signal_stats(train_files_by_folder: dict = None, max_files_per_class: int = 10):
    """Calcula média e desvio padrão GLOBAIS sobre os sinais do Paderborn (estritamente partição de treino)."""
    print("[Z-SCORE] Calculando estatísticas globais do Paderborn (apenas partição de treino)...")
    if train_files_by_folder is None:
        pad_raw_splits = get_paderborn_strict_splits(num_train_files=8, num_val_files=4, num_test_files=5)
        train_files_by_folder = {f: pad_raw_splits["train"][c] for f, c in PADERBORN_FOLDER_MAP.items()}

    all_vals = []
    for folder_name in PADERBORN_FOLDER_MAP.keys():
        mat_files = train_files_by_folder.get(folder_name, [])
        for mat_file in mat_files:
            sig = load_paderborn_mat_file(mat_file)
            all_vals.append(sig)
    all_vals = np.concatenate(all_vals)
    mean, std = float(np.mean(all_vals)), float(np.std(all_vals))
    print(f"    Paderborn global (treino): mean={mean:.6f}, std={std:.6f} ({len(all_vals)} amostras)")
    return mean, std


def _compute_xjtu_global_signal_stats(train_files_by_class: dict = None, max_files_per_class: int = 10):
    """Calcula média e desvio padrão GLOBAIS sobre os sinais do XJTU-SY (estritamente partição de treino)."""
    print("[Z-SCORE] Calculando estatísticas globais do XJTU-SY (apenas partição de treino)...")
    if train_files_by_class is None:
        train_files_by_class = get_xjtu_strict_splits(num_train_files=16, num_val_files=12, num_test_files=10)["train"]

    all_vals = []
    for cls_name in XJTU_CLASSES:
        files = train_files_by_class.get(cls_name, [])
        for csv_file in files:
            sig = load_xjtu_csv_file(csv_file, channel=0)
            all_vals.append(sig)
    all_vals = np.concatenate(all_vals)
    mean, std = float(np.mean(all_vals)), float(np.std(all_vals))
    print(f"    XJTU-SY global (treino): mean={mean:.6f}, std={std:.6f} ({len(all_vals)} amostras)")
    return mean, std


def _cwru_order_cwt_pass1(global_mean, global_std, train_files_by_class: dict = None, samples_per_file: int = 4):
    """Calcula vmin/vmax globais amostrando chunks representativos do CWRU (estritamente partição de treino)."""
    print("[PASS 1] Estimando vmin/vmax globais do CWRU Order-CWT (treino)...")
    if train_files_by_class is None:
        train_files_by_class = get_cwru_strict_splits()["train"]

    all_p1, all_p99 = [], []

    for cls_name in CWRU_CLASSES:
        mat_files = train_files_by_class.get(cls_name, [])
        for mat_file in mat_files:
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
                all_p1.append(np.percentile(cwt_mat, 1))
                all_p99.append(np.percentile(cwt_mat, 99))

    global_vmin = float(np.percentile(all_p1, 5))
    global_vmax = float(np.percentile(all_p99, 95))
    print(f"    CWRU Order-CWT estimado: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
    return global_vmin, global_vmax


def _paderborn_order_cwt_pass1(global_mean, global_std, train_files_by_folder: dict = None, max_files_per_class: int = 10, samples_per_file: int = 4):
    """Calcula vmin/vmax globais amostrando chunks representativos do Paderborn (estritamente partição de treino)."""
    print("[PASS 1] Estimando vmin/vmax globais do Paderborn Order-CWT (treino)...")
    if train_files_by_folder is None:
        pad_raw_splits = get_paderborn_strict_splits(num_train_files=8, num_val_files=4, num_test_files=5)
        train_files_by_folder = {f: pad_raw_splits["train"][c] for f, c in PADERBORN_FOLDER_MAP.items()}

    all_p1, all_p99 = [], []

    for folder_name in PADERBORN_FOLDER_MAP.keys():
        mat_files = train_files_by_folder.get(folder_name, [])
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
                all_p1.append(np.percentile(cwt_mat, 1))
                all_p99.append(np.percentile(cwt_mat, 99))

    global_vmin = float(np.percentile(all_p1, 5))
    global_vmax = float(np.percentile(all_p99, 95))
    print(f"    Paderborn Order-CWT estimado: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
    return global_vmin, global_vmax


def _xjtu_order_cwt_pass1(global_mean, global_std, train_files_by_class: dict = None, max_files_per_class: int = 10, samples_per_file: int = 4):
    """Calcula vmin/vmax globais amostrando chunks representativos do XJTU-SY (estritamente partição de treino)."""
    print("[PASS 1] Estimando vmin/vmax globais do XJTU-SY Order-CWT (treino)...")
    if train_files_by_class is None:
        train_files_by_class = get_xjtu_strict_splits(num_train_files=16, num_val_files=12, num_test_files=10)["train"]

    all_p1, all_p99 = [], []

    for cls_name in XJTU_CLASSES:
        files = train_files_by_class.get(cls_name, [])
        for csv_file in files:
            rpm = extract_xjtu_rpm(csv_file)
            sig = load_xjtu_csv_file(csv_file, channel=0)
            sig = (sig - global_mean) / global_std

            fr = rpm / 60.0
            samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * XJTU_FS))
            max_starts = len(sig) - samples_per_window_time
            if max_starts <= 0:
                continue

            starts = np.linspace(0, max_starts, samples_per_file, dtype=int)
            for start in starts:
                chunk = sig[start: min(start + samples_per_window_time + 8, len(sig))]
                ang_sig = resample_to_angle_domain(
                    chunk, XJTU_FS, rpm,
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
    print(f"    XJTU-SY Order-CWT estimado: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
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
    test_ratio: float = 0.15,
    splits_by_class: dict = None
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

    if splits_by_class is None:
        cwru_splits_raw = get_cwru_strict_splits()
        splits_by_class = {cls: {sp: cwru_splits_raw[sp][cls] for sp in ["train", "val", "test"]} for cls in CWRU_CLASSES}

    for cls_name in CWRU_CLASSES:
        splits = splits_by_class[cls_name]

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

            print(f"    [OK] CWRU {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens CWRU geradas em: {ORDER_CWRU_DIR} ({total_images} imagens)")


def generate_paderborn_order_dataset(
    global_mean: float,
    global_std: float,
    global_vmin: float,
    global_vmax: float,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_files_per_class: int = 10,
    splits_by_folder: dict = None
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

    if splits_by_folder is None:
        pad_splits_raw = get_paderborn_strict_splits(num_train_files=8, num_val_files=4, num_test_files=5)
        splits_by_folder = {
            folder: {sp: pad_splits_raw[sp][cls] for sp in ["train", "val", "test"]}
            for folder, cls in PADERBORN_FOLDER_MAP.items()
        }

    for folder_name, cls_name in PADERBORN_FOLDER_MAP.items():
        splits = splits_by_folder[folder_name]

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

            print(f"    [OK] Paderborn {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens Paderborn geradas em: {ORDER_PADERBORN_DIR} ({total_images} imagens)")


def generate_xjtu_order_dataset(
    global_mean: float,
    global_std: float,
    global_vmin: float,
    global_vmax: float,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_files_per_class: int = 30,
    splits_by_class: dict = None
):
    """Gera escalogramas de ordens para o dataset XJTU-SY com normalização GLOBAL."""
    print("\n" + "=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — XJTU-SY [GLOBAL NORM]")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
    print(f"    Z-Score: mean={global_mean:.4f}, std={global_std:.4f}")
    print(f"    Colormap Norm: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        for cls_name in XJTU_CLASSES:
            (ORDER_XJTU_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

    total_images = 0

    if splits_by_class is None:
        xjtu_splits_raw = get_xjtu_strict_splits(num_train_files=16, num_val_files=12, num_test_files=10)
        splits_by_class = {cls: {sp: xjtu_splits_raw[sp][cls] for sp in ["train", "val", "test"]} for cls in XJTU_CLASSES}

    for cls_name in XJTU_CLASSES:
        splits = splits_by_class[cls_name]

        for split_name, s_files in splits.items():
            dest = ORDER_XJTU_DIR / split_name / cls_name
            idx = 0

            for csv_file in s_files:
                rpm = extract_xjtu_rpm(csv_file)
                sig = load_xjtu_csv_file(csv_file, channel=0)

                # Z-score global
                sig = (sig - global_mean) / global_std

                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * XJTU_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start: min(start + samples_per_window_time + 8, len(sig))]
                    ang_sig = resample_to_angle_domain(
                        chunk, XJTU_FS, rpm,
                        samples_per_rev=ORDER_SAMPLES_PER_REV,
                        num_revs=ORDER_NUM_REVS
                    )
                    cwt_mat, _ = compute_order_cwt(
                        ang_sig,
                        samples_per_rev=ORDER_SAMPLES_PER_REV,
                        order_min=ORDER_MIN,
                        order_max=ORDER_MAX
                    )
                    # Normalização GLOBAL compartilhada
                    img = scalogram_to_rgb(cwt_mat, vmin=global_vmin, vmax=global_vmax)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    [OK] XJTU-SY {cls_name:>10} ({split_name:>5}): {len(s_files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens XJTU-SY geradas em: {ORDER_XJTU_DIR} ({total_images} imagens)")


def generate_all_order_datasets(max_files_per_class: int = 10, include_xjtu: bool = True):
    """
    Pipeline completo de geração dos datasets Order-CWT para CWRU, Paderborn e XJTU-SY:
    1. Pré-particionamento estrito de arquivos (Leave-One-Load-Out, Leave-Operating-Condition-Out, Leave-Bearing-Out).
    2. Z-Score Global dos sinais 1D estimado estritamente sobre a partição de TREINO.
    3. Estimativa Amostrada de vmin/vmax globais compartilhados (Passo 1) sobre a partição de TREINO.
    4. Geração e gravação dos escalogramas PNG com normalização global unificada.
    """
    print("=" * 70)
    print("[PIPELINE ORDER-CWT] Particionamento prévio estrito (Prevenção de Data Leakage)...")
    print("=" * 70)

    # 1. Particionamento estrito de arquivos sem data leakage
    # CWRU: Leave-One-Load-Out (0HP+1HP -> 2HP -> 3HP)
    cwru_splits_raw = get_cwru_strict_splits()
    cwru_splits = {cls: {sp: cwru_splits_raw[sp][cls] for sp in ["train", "val", "test"]} for cls in CWRU_CLASSES}
    cwru_train_files = {cls: cwru_splits[cls]["train"] for cls in CWRU_CLASSES}

    # Paderborn: Leave-Operating-Condition-Out (N15_M07_F10 -> N15_M07_F04 -> N15_M01_F10)
    pad_splits_raw = get_paderborn_strict_splits(num_train_files=8, num_val_files=4, num_test_files=5)
    pad_splits = {
        folder: {sp: pad_splits_raw[sp][cls] for sp in ["train", "val", "test"]}
        for folder, cls in PADERBORN_FOLDER_MAP.items()
    }
    pad_train_files = {folder: pad_splits[folder]["train"] for folder in PADERBORN_FOLDER_MAP.keys()}

    # XJTU-SY: Leave-Bearing-Out com 9 rolamentos físicos exclusivos e falhas puras (Wang et al., 2020)
    xjtu_splits = {}
    xjtu_train_files = {}
    if include_xjtu:
        xjtu_splits_raw = get_xjtu_strict_splits(num_train_files=16, num_val_files=12, num_test_files=10)
        xjtu_splits = {cls: {sp: xjtu_splits_raw[sp][cls] for sp in ["train", "val", "test"]} for cls in XJTU_CLASSES}
        xjtu_train_files = {cls: xjtu_splits[cls]["train"] for cls in XJTU_CLASSES}

    # 2. Estatísticas Z-Score estritamente sobre a partição de treino
    cwru_mean, cwru_std = _compute_cwru_global_signal_stats(train_files_by_class=cwru_train_files)
    pad_mean, pad_std = _compute_paderborn_global_signal_stats(train_files_by_folder=pad_train_files, max_files_per_class=max_files_per_class)

    # 3. Pass 1 para estimativa de vmin/vmax estritamente sobre a partição de treino
    cwru_vmin, cwru_vmax = _cwru_order_cwt_pass1(cwru_mean, cwru_std, train_files_by_class=cwru_train_files, samples_per_file=4)
    pad_vmin, pad_vmax = _paderborn_order_cwt_pass1(pad_mean, pad_std, train_files_by_folder=pad_train_files, samples_per_file=4)

    all_vmins = [cwru_vmin, pad_vmin]
    all_vmaxs = [cwru_vmax, pad_vmax]

    if include_xjtu:
        xjtu_mean, xjtu_std = _compute_xjtu_global_signal_stats(train_files_by_class=xjtu_train_files, max_files_per_class=max_files_per_class)
        xjtu_vmin, xjtu_vmax = _xjtu_order_cwt_pass1(xjtu_mean, xjtu_std, train_files_by_class=xjtu_train_files, samples_per_file=4)
        all_vmins.append(xjtu_vmin)
        all_vmaxs.append(xjtu_vmax)

    shared_vmin = min(all_vmins)
    shared_vmax = max(all_vmaxs)
    print(f"\n[GLOBAL COMPARTILHADO TRIPARTITE (TREINO)] vmin={shared_vmin:.6f}, vmax={shared_vmax:.6f}\n")

    # 4. Geração dos escalogramas com normalização global e splits preservados
    generate_cwru_order_dataset(cwru_mean, cwru_std, shared_vmin, shared_vmax, splits_by_class=cwru_splits)
    generate_paderborn_order_dataset(pad_mean, pad_std, shared_vmin, shared_vmax, max_files_per_class=max_files_per_class, splits_by_folder=pad_splits)
    if include_xjtu:
        generate_xjtu_order_dataset(xjtu_mean, xjtu_std, shared_vmin, shared_vmax, max_files_per_class=30, splits_by_class=xjtu_splits)


if __name__ == "__main__":
    generate_all_order_datasets(max_files_per_class=10, include_xjtu=True)

