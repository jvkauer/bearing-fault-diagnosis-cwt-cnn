"""
Script de Geração de Datasets de Escalogramas Ângulo-Ordem (Order-CWT) para CWRU, Paderborn e XJTU-SY.
Aplica Computed Order Tracking (COT) com reamostragem angular invariante à rotação (RPM)
e normalização local por percentil (1-99%), garantindo paridade metodológica com a CWT de tempo
e divisão estrita por arquivos brutos para evitar Data Leakage.

Melhorias:
- Remoção do offset DC por janela antes da reamostragem angular (chunk - chunk.mean()).
- Normalização local per-sample (scalogram_to_rgb) para preservação total do contraste dinâmico.
- Faixa estendida de ordens (0.5 a 20.0 ordens) para cobertura de harmônicos até 3x BPFI.
"""

import sys
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CLASSES as CWRU_CLASSES,
    PADERBORN_CLASSES,
    PADERBORN_FOLDER_MAP,
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
from src.dataset_xjtu import load_xjtu_csv_file, extract_xjtu_rpm, get_xjtu_strict_splits
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
    name = Path(file_name).name
    if name.startswith("N09"):
        return 900.0
    elif name.startswith("N12"):
        return 1200.0
    elif name.startswith("N15"):
        return 1500.0
    elif name.startswith("N04"):
        return 400.0
    raise ValueError(f"Não foi possível identificar o RPM no nome do arquivo Paderborn: {file_name}")


def generate_cwru_order_dataset(*args, splits_by_class: dict = None, **kwargs):
    """Gera escalogramas de ordens para o dataset CWRU com normalização local (per-sample)."""
    print("=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — CWRU [NORMALIZAÇÃO LOCAL]")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
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

                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * CWRU_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start: min(start + samples_per_window_time + 8, len(sig))]
                    # Remoção de offset DC por janela
                    chunk = chunk - np.mean(chunk)

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
                    # Normalização LOCAL (1%-99% percentile per-sample)
                    img = scalogram_to_rgb(cwt_mat)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    [OK] CWRU {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens CWRU geradas em: {ORDER_CWRU_DIR} ({total_images} imagens)")


def generate_paderborn_order_dataset(*args, max_files_per_class: int = 10, splits_by_folder: dict = None, **kwargs):
    """Gera escalogramas de ordens para o dataset Paderborn com normalização local (per-sample)."""
    print("\n" + "=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — PADERBORN [NORMALIZAÇÃO LOCAL]")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
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

                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * PADERBORN_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start: min(start + samples_per_window_time + 8, len(sig))]
                    # Remoção de offset DC por janela
                    chunk = chunk - np.mean(chunk)

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
                    # Normalização LOCAL (1%-99% percentile per-sample)
                    img = scalogram_to_rgb(cwt_mat)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    [OK] Paderborn {cls_name:>10} ({split_name:>5}): {len(files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens Paderborn geradas em: {ORDER_PADERBORN_DIR} ({total_images} imagens)")


def generate_xjtu_order_dataset(*args, max_files_per_class: int = 30, splits_by_class: dict = None, **kwargs):
    """Gera escalogramas de ordens para o dataset XJTU-SY com normalização local (per-sample)."""
    print("\n" + "=" * 70)
    print(" GERAÇÃO DO DATASET ÂNGULO-ORDEM (ORDER-CWT) — XJTU-SY [NORMALIZAÇÃO LOCAL]")
    print(f"    {ORDER_SAMPLES_PER_REV} amostras/volta | Janela: {ORDER_NUM_REVS} voltas | Ordens: {ORDER_MIN}-{ORDER_MAX}")
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

                fr = rpm / 60.0
                samples_per_window_time = int(np.ceil((ORDER_NUM_REVS / fr) * XJTU_FS))
                step_time = samples_per_window_time // 2

                for start in range(0, len(sig) - samples_per_window_time + 1, step_time):
                    chunk = sig[start: min(start + samples_per_window_time + 8, len(sig))]
                    # Remoção de offset DC por janela
                    chunk = chunk - np.mean(chunk)

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
                    # Normalização LOCAL (1%-99% percentile per-sample)
                    img = scalogram_to_rgb(cwt_mat)
                    img.save(str(dest / f"{cls_name}_{split_name}_{idx:05d}.png"), format="PNG")
                    idx += 1
                    total_images += 1

            print(f"    [OK] XJTU-SY {cls_name:>10} ({split_name:>5}): {len(s_files)} arquivos -> {idx} imagens geradas")

    print(f"\n[+] Total de imagens XJTU-SY geradas em: {ORDER_XJTU_DIR} ({total_images} imagens)")


def generate_all_order_datasets(max_files_per_class: int = 10, include_xjtu: bool = True):
    """
    Pipeline unificado de geração dos datasets Order-CWT para CWRU, Paderborn e XJTU-SY:
    1. Pré-particionamento estrito de arquivos (Leave-One-Load-Out, Leave-Operating-Condition-Out, Leave-Bearing-Out).
    2. Remoção de offset DC por janela.
    3. Reamostragem angular invariante (COT).
    4. Transformada Wavelet no domínio de ordens (0.5 a 20.0 ordens).
    5. Normalização local per-sample (1%-99%), garantindo paridade metodológica com a CWT de tempo.
    """
    print("=" * 70)
    print("[PIPELINE ORDER-CWT] Geração com Normalização Local e Remoção de DC...")
    print("=" * 70)

    # 1. Particionamento estrito de arquivos sem data leakage
    cwru_splits_raw = get_cwru_strict_splits()
    cwru_splits = {cls: {sp: cwru_splits_raw[sp][cls] for sp in ["train", "val", "test"]} for cls in CWRU_CLASSES}

    pad_splits_raw = get_paderborn_strict_splits(num_train_files=8, num_val_files=4, num_test_files=5)
    pad_splits = {
        folder: {sp: pad_splits_raw[sp][cls] for sp in ["train", "val", "test"]}
        for folder, cls in PADERBORN_FOLDER_MAP.items()
    }

    xjtu_splits = {}
    if include_xjtu:
        xjtu_splits_raw = get_xjtu_strict_splits(num_train_files=16, num_val_files=12, num_test_files=10)
        xjtu_splits = {cls: {sp: xjtu_splits_raw[sp][cls] for sp in ["train", "val", "test"]} for cls in XJTU_CLASSES}

    # 2. Geração direta em passagem única com normalização local
    generate_cwru_order_dataset(splits_by_class=cwru_splits)
    generate_paderborn_order_dataset(max_files_per_class=max_files_per_class, splits_by_folder=pad_splits)
    if include_xjtu:
        generate_xjtu_order_dataset(max_files_per_class=30, splits_by_class=xjtu_splits)

    print("\n" + "=" * 70)
    print(" [CONCLUÍDO] Todos os datasets de ordens foram gerados com sucesso!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_order_datasets(max_files_per_class=10, include_xjtu=True)
