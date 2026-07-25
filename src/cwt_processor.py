"""
Módulo de Processamento CWT (Continuous Wavelet Transform).
Converte janelas de sinais de vibração unidimensionais em escalogramas bidimensionais.
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
from src.config import FS, WAVELET, FREQ_MIN, FREQ_MAX, IMG_HEIGHT, IMG_WIDTH


def get_scales():
    """
    Gera as escalas da wavelet correspondentes às frequências linearmente espaçadas
    entre FREQ_MIN e FREQ_MAX.
    
    Returns:
        tuple: (escalas, frequencias_desejadas)
    """
    freqs_desejadas = np.linspace(FREQ_MIN, FREQ_MAX, IMG_HEIGHT)
    escalas = pywt.frequency2scale(WAVELET, freqs_desejadas / FS)
    return escalas, freqs_desejadas


def compute_cwt(signal_window: np.ndarray) -> np.ndarray:
    """
    Calcula a Transformada Wavelet Contínua (CWT) para uma janela de sinal.

    Args:
        signal_window (np.ndarray): Janela temporal do sinal (1D array de tamanho 1024).

    Returns:
        np.ndarray: Matriz 2D de magnitude do escalograma (shape: IMG_HEIGHT, len(signal_window)).
    """
    escalas, _ = get_scales()
    coefs_cwt, _ = pywt.cwt(
        signal_window,
        escalas,
        WAVELET,
        sampling_period=1.0 / FS,
        method="conv"
    )
    # Extrair a magnitude dos coeficientes complexos
    scalogram_magnitude = np.abs(coefs_cwt)
    return scalogram_magnitude


def scalogram_to_rgb(scalogram_magnitude: np.ndarray, cmap_name: str = "jet") -> Image.Image:
    """
    Converte a matriz numérica do escalograma em uma Imagem PIL RGB (224x224),
    sem bordas, eixos ou elementos visuais.

    Args:
        scalogram_magnitude (np.ndarray): Matriz de magnitude CWT.
        cmap_name (str): Nome do colormap do matplotlib (padrão: 'jet').

    Returns:
        PIL.Image.Image: Imagem RGB redimensionada para (IMG_WIDTH, IMG_HEIGHT).
    """
    # Normalizar valores para intervalo [0, 1]
    norm = plt.Normalize(vmin=scalogram_magnitude.min(), vmax=scalogram_magnitude.max())
    cmap = plt.get_cmap(cmap_name)
    
    # Aplicar colormap -> array RGBA (0..1)
    rgba_mat = cmap(norm(scalogram_magnitude))
    
    # Converter para uint8 RGB (0..255)
    rgb_mat = (rgba_mat[:, :, :3] * 255).astype(np.uint8)
    
    # Criar Imagem PIL
    img = Image.fromarray(rgb_mat)
    
    # Redimensionar para a dimensão exata da CNN (224x224)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    return img


def save_scalogram_png(scalogram_magnitude: np.ndarray, output_path: str, cmap_name: str = "jet"):
    """
    Salva diretamente a matriz do escalograma como arquivo PNG RGB de 224x224 pixels.

    Args:
        scalogram_magnitude (np.ndarray): Matriz de magnitude CWT.
        output_path (str): Caminho onde o arquivo PNG será salvo.
        cmap_name (str): Colormap a utilizar.
    """
    img = scalogram_to_rgb(scalogram_magnitude, cmap_name=cmap_name)
    img.save(output_path, format="PNG")
