"""
Módulo de Computed Order Tracking (COT) e Transformada Wavelet Ângulo-Ordem (Order-CWT).
Transforma sinais temporais em sinais amostrados no domínio angular (voltas do eixo),
garantindo invariância à velocidade de rotação (RPM) para diagnóstico de falhas em rolamentos.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import pywt
import matplotlib.pyplot as plt

from src.config import WAVELET, IMG_HEIGHT


def resample_to_angle_domain(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    samples_per_rev: int = 1024,
    num_revs: int = 4
) -> np.ndarray:
    """
    Reamostra um sinal temporal para o domínio angular (amostras uniformes por volta do eixo).

    Args:
        signal (np.ndarray): Sinal de vibração temporal 1D.
        fs (float): Taxa de amostragem original em Hz.
        rpm (float): Velocidade de rotação do eixo em RPM.
        samples_per_rev (int): Número de amostras por 1 rotação completa (padrão 1024).
        num_revs (int): Número de rotações do eixo a serem extraídas na janela (padrão 4 voltas).

    Returns:
        np.ndarray: Sinal 1D reamostrado no domínio angular de tamanho (samples_per_rev * num_revs).
    """
    # Frequência fundamental do eixo em Hz e período de 1 volta em segundos
    fr = rpm / 60.0
    t_rev = 1.0 / fr
    
    # Tempo total necessário para num_revs voltas
    total_time = num_revs * t_rev
    total_time_samples = int(np.ceil(total_time * fs))
    
    if len(signal) < total_time_samples:
        raise ValueError(
            f"Sinal muito curto ({len(signal)} amostras) para {num_revs} voltas a {rpm} RPM ({total_time_samples} amostras necessárias)."
        )
        
    # Adicionar margem de segurança para evitar extrapolação cúbica nos extremos
    safe_samples = min(total_time_samples + 4, len(signal))
    signal_segment = signal[:safe_samples]
    t_orig = np.arange(len(signal_segment)) / fs
    
    # Novo vetor temporal com amostragem angular uniforme
    # Garantir que t_angular nunca ultrapasse o último ponto de t_orig
    total_angle_samples = samples_per_rev * num_revs
    t_angular = np.linspace(0, total_time, total_angle_samples, endpoint=False)
    
    # Clip para garantir que nenhum ponto de t_angular exceda t_orig[-1]
    t_angular = np.clip(t_angular, t_orig[0], t_orig[-1])
    
    # Interpolação cúbica com limites seguros (sem extrapolação polinomial)
    interpolator = interp1d(
        t_orig, signal_segment, kind="cubic",
        bounds_error=False,
        fill_value=(signal_segment[0], signal_segment[-1])
    )
    angle_signal = interpolator(t_angular)
    
    return angle_signal


def compute_order_cwt(
    angle_signal: np.ndarray,
    samples_per_rev: int = 1024,
    order_min: float = 0.5,
    order_max: float = 15.0,
    num_scales: int = IMG_HEIGHT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula a Transformada Wavelet Contínua no Domínio de Ordens (Order-CWT).

    Args:
        angle_signal (np.ndarray): Sinal amostrado no domínio angular (amostras uniformes por volta).
        samples_per_rev (int): Taxa de amostragem angular (amostras/revolução).
        order_min (float): Ordem mínima de interesse (múltiplos da rotação do eixo).
        order_max (float): Ordem máxima de interesse (múltiplos da rotação do eixo).
        num_scales (int): Resolução vertical do escalograma (padrão 224 ordens).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Matriz de magnitude do escalograma de ordens, vetor de ordens).
    """
    target_orders = np.linspace(order_min, order_max, num_scales)
    scales = pywt.frequency2scale(WAVELET, target_orders / samples_per_rev)
    
    coefs_cwt, _ = pywt.cwt(
        angle_signal,
        scales,
        WAVELET,
        sampling_period=1.0 / samples_per_rev,
        method="conv"
    )
    
    return np.abs(coefs_cwt), target_orders


def plot_time_vs_order_comparison(
    sig_cwru: np.ndarray,
    rpm_cwru: float,
    fs_cwru: float,
    sig_pad: np.ndarray,
    rpm_pad: float,
    fs_pad: float,
    fault_name: str = "Pista Interna (Inner Race)",
    save_path: Optional[str] = None
):
    """
    Gera figura comparativa lado a lado demonstrando a superação do Domain Shift
    através do Order Tracking (Tempo-Frequência vs. Ângulo-Ordem).
    """
    # 1. CWT Tradicional no Tempo
    win_cwru = sig_cwru[:1024]
    scales_cwru = pywt.frequency2scale(WAVELET, np.linspace(10, 6000, 224) / fs_cwru)
    cwt_time_cwru, _ = pywt.cwt(win_cwru, scales_cwru, WAVELET, sampling_period=1.0/fs_cwru, method="conv")
    
    win_pad = sig_pad[:4096]
    scales_pad = pywt.frequency2scale(WAVELET, np.linspace(10, 20000, 224) / fs_pad)
    cwt_time_pad, _ = pywt.cwt(win_pad, scales_pad, WAVELET, sampling_period=1.0/fs_pad, method="conv")
    
    # 2. CWT com Order Tracking (Ângulo-Ordem)
    ang_cwru = resample_to_angle_domain(sig_cwru, fs_cwru, rpm_cwru, samples_per_rev=1024, num_revs=4)
    cwt_order_cwru, orders = compute_order_cwt(ang_cwru, samples_per_rev=1024, order_min=0.5, order_max=15.0)
    
    ang_pad = resample_to_angle_domain(sig_pad, fs_pad, rpm_pad, samples_per_rev=1024, num_revs=4)
    cwt_order_pad, _ = compute_order_cwt(ang_pad, samples_per_rev=1024, order_min=0.5, order_max=15.0)
    
    # 3. Montar Painel Comparativo 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # Linha 1: Domínio Tradicional do Tempo (Desalinhado pelo RPM)
    t_c = np.arange(len(win_cwru)) / fs_cwru * 1000
    axes[0, 0].imshow(np.abs(cwt_time_cwru), extent=[t_c[0], t_c[-1], 10, 6000], cmap="jet", aspect="auto", origin="lower")
    axes[0, 0].set_title(f"CWRU: Tempo-Frequência ({rpm_cwru:.0f} RPM)\n[Frequência de Falha ~160 Hz]", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Tempo (ms)")
    axes[0, 0].set_ylabel("Frequência (Hz)")
    
    t_p = np.arange(len(win_pad)) / fs_pad * 1000
    axes[0, 1].imshow(np.abs(cwt_time_pad), extent=[t_p[0], t_p[-1], 10, 20000], cmap="jet", aspect="auto", origin="lower")
    axes[0, 1].set_title(f"Paderborn: Tempo-Frequência ({rpm_pad:.0f} RPM)\n[Frequência de Falha ~74 Hz]", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Tempo (ms)")
    axes[0, 1].set_ylabel("Frequência (Hz)")
    
    # Linha 2: Domínio Ângulo-Ordem (Alinhado Perfeitamente pelo Order Tracking!)
    revs = np.linspace(0, 4, len(ang_cwru))
    axes[1, 0].imshow(cwt_order_cwru, extent=[revs[0], revs[-1], 0.5, 15.0], cmap="jet", aspect="auto", origin="lower")
    axes[1, 0].set_title(f"CWRU: Ângulo-Ordem ({rpm_cwru:.0f} RPM)\n[Ordem de Falha BPFI ~ 5.4 Ordens]", fontsize=11, fontweight="bold", color="darkgreen")
    axes[1, 0].set_xlabel("Rotação do Eixo (Voltas)")
    axes[1, 0].set_ylabel("Ordens (f / fr)")
    axes[1, 0].axhline(5.4, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    
    axes[1, 1].imshow(cwt_order_pad, extent=[revs[0], revs[-1], 0.5, 15.0], cmap="jet", aspect="auto", origin="lower")
    axes[1, 1].set_title(f"Paderborn: Ângulo-Ordem ({rpm_pad:.0f} RPM)\n[Ordem de Falha BPFI ~ 4.9 Ordens]", fontsize=11, fontweight="bold", color="darkgreen")
    axes[1, 1].set_xlabel("Rotação do Eixo (Voltas)")
    axes[1, 1].set_ylabel("Ordens (f / fr)")
    axes[1, 1].axhline(4.9, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    
    fig.suptitle(f"Invariância à Velocidade via Computed Order Tracking (COT) — {fault_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Gráfico comparativo de Order Tracking salvo em: {save_path}")
        
    plt.show()
