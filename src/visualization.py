"""
Módulo de Visualização de Sinais e Escalogramas.
Gera figuras no formato da Monografia de TCC (Figura 5):
  (a) Sinal Bruto no Tempo
  (b) Escalograma CWT (Morlet Complexa)
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from src.config import FS, WAVELET, FREQ_MIN, FREQ_MAX, IMG_HEIGHT
from src.cwt_processor import compute_cwt, get_scales


def plot_monograph_figure(
    signal_window: np.ndarray,
    title: str = "Diagnóstico de Falha em Rolamento",
    save_path: str = None
):
    """
    Gera e exibe o gráfico no padrão da Figura 5 da Monografia:
    - Painel (a): Sinal Bruto (Tempo em segundos vs Amplitude em g)
    - Painel (b): Escalograma CWT (Tempo em segundos vs Frequência em Hz)

    Args:
        signal_window (np.ndarray): Janela do sinal de vibração (1D).
        title (str): Título principal do gráfico.
        save_path (str, optional): Caminho para salvar a figura em PNG/PDF.
    """
    vetor_tempo = np.arange(len(signal_window)) / FS
    freqs_desejadas = np.linspace(FREQ_MIN, FREQ_MAX, IMG_HEIGHT)
    
    # Calcular magnitude da CWT
    scalogram_mat = compute_cwt(signal_window)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # --------------------------------------------------------------------------
    # Subplot (a): Sinal no Tempo
    # --------------------------------------------------------------------------
    axs[0].plot(vetor_tempo, signal_window, color="#3182bd", linewidth=0.8)
    axs[0].set_title("(a) Sinal Bruto", fontsize=11, fontweight="bold")
    axs[0].set_ylabel("Amplitude (g)", fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    
    # --------------------------------------------------------------------------
    # Subplot (b): Escalograma CWT (suave para visualização na monografia)
    # --------------------------------------------------------------------------
    pcm = axs[1].imshow(
        scalogram_mat,
        extent=[vetor_tempo[0], vetor_tempo[-1], freqs_desejadas[0], freqs_desejadas[-1]],
        cmap="jet",
        aspect="auto",
        origin="lower",
        interpolation="bilinear"  # Suavização apenas para visualização humana
    )
    
    axs[1].set_title("(b) Scalogram CWT (Morlet Complexa)", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Tempo (s)", fontsize=10)
    axs[1].set_ylabel("Frequência (Hz)", fontsize=10)
    
    # Colorbar
    fig.colorbar(pcm, ax=axs[1], label="Magnitude |C(a,b)|")
    
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura salva com sucesso em: {save_path}")
        
    plt.show()
