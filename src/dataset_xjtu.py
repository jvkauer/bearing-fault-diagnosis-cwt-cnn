"""
Módulo de Carregamento e Janelamento do Dataset XJTU-SY (Xi'an Jiaotong University).
Processa os arquivos .csv de 25.6 kHz e extrai o sinal do acelerômetro horizontal.
Implementa o mapeamento estrito das classes (Normal, Pista Interna e Pista Externa).
"""

from pathlib import Path
from typing import List, Dict
import numpy as np

from src.config import (
    XJTU_RAW_DIR,
    XJTU_CLASSES,
    XJTU_FS,
    XJTU_WINDOW_SIZE,
    XJTU_STEP_SIZE
)


def load_xjtu_csv_file(file_path: Path, channel: int = 0) -> np.ndarray:
    """
    Carrega um arquivo .csv do Dataset XJTU-SY e retorna o canal de aceleração especificado.

    Args:
        file_path (Path): Caminho completo para o arquivo .csv.
        channel (int): Índice da coluna (0 = horizontal, 1 = vertical). Padrão: 0 (horizontal).

    Returns:
        np.ndarray: Sinal 1D de aceleração amostrado a 25.6 kHz.
    """
    data = np.loadtxt(str(file_path), delimiter=",", skiprows=1, usecols=channel, dtype=np.float64)
    return data


def extract_xjtu_rpm(file_path: Path) -> float:
    """
    Identifica a rotação do motor (RPM) a partir do caminho do arquivo XJTU-SY.

    Condições operacionais:
        - 35Hz12kN:   35 Hz * 60 = 2100 RPM
        - 37.5Hz11kN: 37.5 Hz * 60 = 2250 RPM
        - 40Hz10kN:   40 Hz * 60 = 2400 RPM
    """
    path_str = str(file_path).replace("\\", "/")
    if "35Hz" in path_str:
        return 2100.0
    elif "37.5Hz" in path_str:
        return 2250.0
    elif "40Hz" in path_str:
        return 2400.0
    return 2100.0


def segment_xjtu_signal(
    signal: np.ndarray,
    window_size: int = XJTU_WINDOW_SIZE,
    step_size: int = XJTU_STEP_SIZE
) -> List[np.ndarray]:
    """
    Segmenta o sinal 1D de vibração de 25.6 kHz em janelas temporais usando janela deslizante.

    Args:
        signal (np.ndarray): Sinal de vibração contínuo (32.768 amostras no XJTU-SY).
        window_size (int): Tamanho da janela (padrão: 2048 amostras ~80 ms).
        step_size (int): Passo de deslocamento (padrão: 1024 amostras / 50% overlap).

    Returns:
        List[np.ndarray]: Lista de arrays 1D contendo cada janela.
    """
    windows = []
    n_samples = len(signal)
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])
    return windows


def _sort_csvs_numerically(folder: Path) -> List[Path]:
    """Retorna os arquivos .csv de uma pasta ordenados numericamente (1.csv, 2.csv, ...)."""
    if not folder.exists():
        return []
    files = list(folder.glob("*.csv"))
    return sorted(files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def get_xjtu_manifest(max_files_per_class: int = 30) -> Dict[str, List[Path]]:
    """
    Monta o manifesto de arquivos por classe para o dataset XJTU-SY.

    Regras de seleção baseadas na evolução acelerada de falhas (Run-to-failure):
      - 'normal': Arquivos iniciais (1.csv a 25.csv) de rolamentos novos em teste.
      - 'outer_race': Últimos arquivos de rolamentos com falha comprovada na pista externa.
      - 'inner_race': Últimos arquivos de rolamentos com falha comprovada na pista interna.

    Args:
        max_files_per_class (int): Quantidade balanceada de arquivos por classe.

    Returns:
        Dict[str, List[Path]]: Dicionário mapeando cada classe à sua lista de arquivos CSV.
    """
    manifest = {cls: [] for cls in XJTU_CLASSES}

    # -------------------------------------------------------------------------
    # 1. CLASSE NORMAL (Arquivos iniciais de rolamentos saudáveis)
    # -------------------------------------------------------------------------
    normal_candidates = []
    for cond, b_name in [
        ("35Hz12kN", "Bearing1_1"),
        ("35Hz12kN", "Bearing1_2"),
        ("37.5Hz11kN", "Bearing2_1"),
        ("37.5Hz11kN", "Bearing2_2")
    ]:
        folder = XJTU_RAW_DIR / cond / b_name
        files = _sort_csvs_numerically(folder)
        # Pega os primeiros 15 arquivos de cada teste inicial
        normal_candidates.extend(files[:15])

    manifest["normal"] = normal_candidates[:max_files_per_class] if max_files_per_class else normal_candidates

    # -------------------------------------------------------------------------
    # 2. CLASSE PISTA EXTERNA (Outer Race - Últimos arquivos com dano severo)
    # -------------------------------------------------------------------------
    outer_candidates = []
    for cond, b_name in [
        ("35Hz12kN", "Bearing1_1"),
        ("35Hz12kN", "Bearing1_2"),
        ("35Hz12kN", "Bearing1_3"),
        ("37.5Hz11kN", "Bearing2_2"),
        ("37.5Hz11kN", "Bearing2_5")
    ]:
        folder = XJTU_RAW_DIR / cond / b_name
        files = _sort_csvs_numerically(folder)
        if files:
            # Pega os últimos 15 arquivos antes da falha total
            outer_candidates.extend(files[-15:])

    manifest["outer_race"] = outer_candidates[:max_files_per_class] if max_files_per_class else outer_candidates

    # -------------------------------------------------------------------------
    # 3. CLASSE PISTA INTERNA (Inner Race - Últimos arquivos com dano severo)
    # -------------------------------------------------------------------------
    inner_candidates = []
    for cond, b_name in [
        ("37.5Hz11kN", "Bearing2_1"),
        ("40Hz10kN", "Bearing3_4")
    ]:
        folder = XJTU_RAW_DIR / cond / b_name
        files = _sort_csvs_numerically(folder)
        if files:
            # Pega os últimos 30 arquivos antes da quebra
            inner_candidates.extend(files[-30:])

    manifest["inner_race"] = inner_candidates[:max_files_per_class] if max_files_per_class else inner_candidates

    return manifest
