"""
Módulo de Carregamento e Janelamento do Dataset de Paderborn (PU Dataset).
Processa os arquivos .mat da Universidade de Paderborn e extrai o sinal de vibração a 64 kHz.
"""

from pathlib import Path
from typing import List
import scipy.io
import numpy as np

from src.config import (
    PADERBORN_RAW_DIR,
    PADERBORN_FOLDER_MAP,
    PADERBORN_WINDOW_SIZE,
    PADERBORN_STEP_SIZE
)


def load_paderborn_mat_file(file_path: Path, channel_name: str = "vibration_1") -> np.ndarray:
    """
    Carrega um arquivo .mat do Dataset de Paderborn e extrai o sinal do canal especificado.

    Args:
        file_path (Path): Caminho para o arquivo .mat do Paderborn.
        channel_name (str): Nome do canal a ser extraído (padrão: 'vibration_1').

    Returns:
        np.ndarray: Sinal unidimensional (1D) de vibração a 64 kHz.
    """
    mat_data = scipy.io.loadmat(str(file_path))
    
    top_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    if not top_keys:
        raise KeyError(f"Nenhuma struct de dados encontrada no arquivo {file_path}")
        
    struct_key = top_keys[0]
    data_struct = mat_data[struct_key]
    
    if "Y" not in data_struct.dtype.names:
        raise KeyError(f"Campo 'Y' não encontrado na struct do arquivo {file_path}")
        
    y_channels = data_struct["Y"][0, 0]
    
    target_data = None
    for idx in range(y_channels.shape[1]):
        ch = y_channels[0, idx]
        ch_name = ch["Name"][0] if len(ch["Name"]) > 0 else ""
        if ch_name == channel_name:
            raw_data = ch["Data"]
            target_data = raw_data[0].flatten().astype(np.float64)
            break
            
    if target_data is None:
        available_names = [y_channels[0, i]["Name"][0] for i in range(y_channels.shape[1]) if len(y_channels[0, i]["Name"]) > 0]
        raise KeyError(f"Canal '{channel_name}' não encontrado. Canais disponíveis: {available_names}")
        
    return target_data


def segment_paderborn_signal(
    signal: np.ndarray,
    window_size: int = PADERBORN_WINDOW_SIZE,
    step_size: int = PADERBORN_STEP_SIZE
) -> List[np.ndarray]:
    """
    Segmenta o sinal 1D de vibração de 64 kHz em janelas temporais usando janela deslizante.

    Args:
        signal (np.ndarray): Sinal de vibração completo (1D).
        window_size (int): Tamanho de cada janela em amostras (padrão: 4096 amostras a 64 kHz).
        step_size (int): Deslocamento entre janelas sucessivas (padrão: 2048 amostras / 50% overlap).

    Returns:
        List[np.ndarray]: Lista de arrays 1D contendo cada janela temporal.
    """
    windows = []
    n_samples = len(signal)
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])
        
    return windows
