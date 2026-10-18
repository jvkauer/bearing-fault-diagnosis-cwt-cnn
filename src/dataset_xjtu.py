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
    raise ValueError(f"Não foi possível identificar o RPM no caminho XJTU-SY: {file_path}")



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


def get_xjtu_strict_splits(
    raw_dir: Path = None,
    num_train_files: int = 16,
    num_val_files: int = 12,
    num_test_files: int = 10
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Retorna a partição estrita de arquivos .csv do XJTU-SY com separação rigorosa de rolamentos
    físicos (Leave-Bearing-Out) e garantia física de falhas puras (Wang et al., 2020 Tabela II).

    Regras Metodológicas:
      1. Zero Rolamentos Compartilhados: Nenhuma identidade física de rolamento aparece em mais
         de uma partição (Treino ∩ Validação = ∅, Treino ∩ Teste = ∅, Validação ∩ Teste = ∅).
      2. Zero Rolamentos Compartilhados entre Classes: Cada classe dentro de cada partição utiliza
         um rolamento físico exclusivo (9 rolamentos físicos distintos no total).
      3. Falhas Puras Comprovadas: Exclusão rigorosa de falhas combinadas (Bearing1_5 e Bearing3_2) e
         de gaiola (Bearing1_4 e Bearing2_3). Todas as amostras de falha interna e externa
         possuem tipo de avaria pura catalogado na literatura oficial.
      4. Linha de Base Normal Reprodutível: Amostras saudáveis extraídas estritamente dos
         primeiros arquivos (fase estacionária nominal pré-degradação).
      5. Condição Inédita no Teste: O conjunto de teste é composto exclusivamente por
         rolamentos da condição 40 Hz / 10 kN (2400 RPM).

    Alocação dos 9 Rolamentos Físicos Exclusivos:
      - Treino:
          * normal:     Bearing1_1 (início saudável - 35 Hz) [16 arquivos]
          * outer_race: Bearing1_2 (falha severa pista externa - 35 Hz) [16 arquivos]
          * inner_race: Bearing2_1 (falha severa pista interna pura - 37.5 Hz) [16 arquivos]
      - Validação:
          * normal:     Bearing2_2 (início saudável - 37.5 Hz) [12 arquivos]
          * outer_race: Bearing2_4 (falha severa pista externa - 37.5 Hz) [12 arquivos]
          * inner_race: Bearing3_3 (falha severa pista interna pura - 40 Hz) [12 arquivos]
      - Teste (Rolamentos e Condição 40 Hz 100% Inéditos no Treino):
          * normal:     Bearing3_1 (início saudável - 40 Hz) [10 arquivos]
          * outer_race: Bearing3_5 (falha severa pista externa pura - 40 Hz) [10 arquivos]
          * inner_race: Bearing3_4 (falha severa pista interna pura - 40 Hz) [10 arquivos]

    Returns:
        Dict[str, Dict[str, List[Path]]]: {"train": {...}, "val": {...}, "test": {...}}
    """
    target_dir = raw_dir if raw_dir is not None else XJTU_RAW_DIR

    def _get_files(cond: str, b_name: str) -> List[Path]:
        return _sort_csvs_numerically(target_dir / cond / b_name)

    # Coletar arquivos por rolamento
    b1_1 = _get_files("35Hz12kN", "Bearing1_1")
    b1_2 = _get_files("35Hz12kN", "Bearing1_2")
    b2_1 = _get_files("37.5Hz11kN", "Bearing2_1")

    b2_2 = _get_files("37.5Hz11kN", "Bearing2_2")
    b2_4 = _get_files("37.5Hz11kN", "Bearing2_4")
    b3_3 = _get_files("40Hz10kN", "Bearing3_3")

    b3_1 = _get_files("40Hz10kN", "Bearing3_1")
    b3_5 = _get_files("40Hz10kN", "Bearing3_5")
    b3_4 = _get_files("40Hz10kN", "Bearing3_4")

    splits = {
        "train": {
            "normal": b1_1[:num_train_files],
            "outer_race": b1_2[-num_train_files:] if b1_2 else [],
            "inner_race": b2_1[-num_train_files:] if b2_1 else [],
        },
        "val": {
            "normal": b2_2[:num_val_files],
            "outer_race": b2_4[-num_val_files:] if b2_4 else [],
            "inner_race": b3_3[-num_val_files:] if b3_3 else [],
        },
        "test": {
            "normal": b3_1[:num_test_files],
            "outer_race": b3_5[-num_test_files:] if b3_5 else [],
            "inner_race": b3_4[-num_test_files:] if b3_4 else [],
        }
    }

    return splits


def get_xjtu_manifest(max_files_per_class: int = 30) -> Dict[str, List[Path]]:
    """
    [EXPLORAÇÃO/INSPEÇÃO] Retorna arquivos XJTU-SY para testes exploratórios.
    ATENÇÃO: Para treinamento e avaliação sem vazamento de dados, utilize
    estritamente 'get_xjtu_strict_splits()', que separa rolamentos disjuntos.

    Args:
        max_files_per_class (int): Quantidade total de arquivos por classe a retornar (padrão: 30).

    Returns:
        Dict[str, List[Path]]: Dicionário mapeando cada classe à sua lista de arquivos CSV.
    """
    splits = get_xjtu_strict_splits()
    manifest = {cls: [] for cls in XJTU_CLASSES}

    for cls_name in XJTU_CLASSES:
        for split_name in ["train", "val", "test"]:
            manifest[cls_name].extend(splits[split_name][cls_name])
        manifest[cls_name] = manifest[cls_name][:max_files_per_class]

    return manifest


