"""
Módulo de Carregamento e Janelamento do Dataset CWRU.
Carrega os arquivos .mat do CWRU e realiza o janelamento deslizante dos sinais de vibração.
"""

from pathlib import Path
import scipy.io
import numpy as np
from src.config import RAW_DATA_DIR, CLASSES, WINDOW_SIZE, STEP_SIZE


def load_cwru_mat_file(file_path: Path) -> np.ndarray:
    """
    Carrega um arquivo .mat do CWRU e extrai o sinal do acelerômetro no Drive End (_DE_time).

    Args:
        file_path (Path): Caminho do arquivo .mat.

    Returns:
        np.ndarray: Sinal unidimensional de vibração (Drive End).
    """
    mat_data = scipy.io.loadmat(str(file_path))
    
    # Procurar a chave que contém o sinal do Drive End (_DE_time)
    de_key = None
    for key in mat_data.keys():
        if key.endswith("_DE_time"):
            de_key = key
            break
            
    if de_key is None:
        raise KeyError(f"Nenhum canal '_DE_time' encontrado no arquivo {file_path}")
        
    signal = mat_data[de_key].flatten()
    return signal


def segment_signal(signal: np.ndarray, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE) -> list[np.ndarray]:
    """
    Segmenta um sinal 1D em janelas temporais utilizando janelamento deslizante (sliding window).

    Args:
        signal (np.ndarray): Sinal de vibração completo.
        window_size (int): Tamanho de cada janela em amostras.
        step_size (int): Deslocamento entre janelas sucessivas.

    Returns:
        list[np.ndarray]: Lista de janelas temporais de tamanho window_size.
    """
    windows = []
    n_samples = len(signal)
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])
        
    return windows


def load_class_windows(class_name: str) -> list[np.ndarray]:
    """
    Carrega todos os arquivos .mat pertencentes a uma determinada classe e extrai todas as janelas.

    Args:
        class_name (str): Nome da classe ('normal', 'inner_race', 'outer_race', 'ball').

    Returns:
        list[np.ndarray]: Lista com todas as janelas extraídas para essa classe.
    """
    class_dir = RAW_DATA_DIR / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Diretório da classe não encontrado: {class_dir}")
        
    mat_files = list(class_dir.glob("*.mat"))
    all_windows = []
    
    for file_path in mat_files:
        signal = load_cwru_mat_file(file_path)
        windows = segment_signal(signal)
        all_windows.extend(windows)
        
    return all_windows
