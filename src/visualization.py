"""
Módulo de Visualização de Sinais, Escalogramas CWT e Curvas de Aprendizado.
Fornece funções para plotagem de figuras acadêmicas (Padrão Monografia TCC) e curvas de treino/validação.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.config import FS, FREQ_MIN, FREQ_MAX, IMG_HEIGHT
from src.cwt_processor import compute_cwt
from sklearn.metrics import confusion_matrix

def plot_monograph_figure(
    signal_window: np.ndarray,
    title: str = "Diagnóstico de Falha em Rolamento",
    save_path: str = None
):
    """
    Gera e exibe o gráfico no padrão da Monografia TCC (Figura 5):
    - Painel (a): Sinal Bruto (Tempo em segundos vs Amplitude em g)
    - Painel (b): Escalograma CWT (Tempo em segundos vs Frequência em Hz)

    Args:
        signal_window (np.ndarray): Janela do sinal de vibração (1D).
        title (str): Título principal do gráfico.
        save_path (str, optional): Caminho para salvar a figura em PNG/PDF.
    """
    time_vector = np.arange(len(signal_window)) / FS
    target_freqs = np.linspace(FREQ_MIN, FREQ_MAX, IMG_HEIGHT)
    
    # Calcular magnitude da CWT
    scalogram_mat = compute_cwt(signal_window)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # --------------------------------------------------------------------------
    # Subplot (a): Sinal no Tempo
    # --------------------------------------------------------------------------
    axs[0].plot(time_vector, signal_window, color="#3182bd", linewidth=0.8)
    axs[0].set_title("(a) Sinal Bruto", fontsize=11, fontweight="bold")
    axs[0].set_ylabel("Amplitude (g)", fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    
    # --------------------------------------------------------------------------
    # Subplot (b): Escalograma CWT (Morlet Complexa)
    # --------------------------------------------------------------------------
    pcm = axs[1].imshow(
        scalogram_mat,
        extent=[time_vector[0], time_vector[-1], target_freqs[0], target_freqs[-1]],
        cmap="jet",
        aspect="auto",
        origin="lower",
        interpolation="bilinear"  # Suavização apenas para visualização humana
    )
    
    axs[1].set_title("(b) Scalogram CWT (Morlet Complexa)", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Tempo (s)", fontsize=10)
    axs[1].set_ylabel("Frequência (Hz)", fontsize=10)
    
    # Barra de cores
    fig.colorbar(pcm, ax=axs[1], label="Magnitude |C(a,b)|")
    
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figura salva em: {save_path}")
        
    plt.show()


def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: str = None
):
    """
    Plota as curvas de aprendizado (Perda e Acurácia) de treino/validação,
    destacando a época do checkpoint (menor val_loss) salvo durante o treino.

    Args:
        history (dict): Dicionário com listas 'train_loss', 'val_loss',
                        'train_acc', 'val_acc' (retornado pelas funções
                        de treino em cnn_processor.py / transfer_learning.py).
        model_name (str): Nome do modelo, usado nos títulos (ex: "ResNet18").
        save_path (str, optional): Caminho para salvar a figura em PNG.
    """
    epochs_axis = range(1, len(history["train_loss"]) + 1)

    # Época do checkpoint = menor val_loss (mesmo critério usado no treino)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = min(history["val_loss"])
    best_val_acc = history["val_acc"][best_epoch - 1]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --------------------------------------------------------------------
    # Painel (a): Curva de Perda
    # --------------------------------------------------------------------
    axs[0].plot(epochs_axis, history["train_loss"],
                label=f"Treino (final: {history['train_loss'][-1]:.4f})",
                marker="o", color="#3182bd", linewidth=1.8)
    axs[0].plot(epochs_axis, history["val_loss"],
                label=f"Validação (final: {history['val_loss'][-1]:.4f})",
                marker="s", color="#e6550d", linewidth=1.8)
    axs[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    axs[0].scatter([best_epoch], [best_val_loss], color="#2ca02c", s=90, zorder=5,
                   marker="*", label=f"Checkpoint (época {best_epoch})")
    axs[0].set_title(f"Curva de Perda (Loss) — {model_name}", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Época")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].legend(fontsize=9, loc="best")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # --------------------------------------------------------------------
    # Painel (b): Curva de Acurácia
    # --------------------------------------------------------------------
    axs[1].plot(epochs_axis, history["train_acc"],
                label=f"Treino (final: {history['train_acc'][-1]:.2f}%)",
                marker="o", color="#3182bd", linewidth=1.8)
    axs[1].plot(epochs_axis, history["val_acc"],
                label=f"Validação (final: {history['val_acc'][-1]:.2f}%)",
                marker="s", color="#e6550d", linewidth=1.8)
    axs[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    axs[1].scatter([best_epoch], [best_val_acc], color="#2ca02c", s=90, zorder=5,
                   marker="*", label=f"Checkpoint (época {best_epoch})")
    axs[1].set_title(f"Curva de Acurácia (%) — {model_name}", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Época")
    axs[1].set_ylabel("Acurácia (%)")
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].legend(fontsize=9, loc="best")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Curvas de Aprendizado — {model_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figura salva em: {save_path}")

    plt.show()

    
def plot_confusion_matrix(
    res: dict,
    model_name: str,
    save_path: str = None
):
    """
    Plota e exibe a Matriz de Confusão para as predições no conjunto de teste.

    Args:
        res (dict): Dicionário contendo 'test_labels', 'test_preds' e 'classes'.
        model_name (str): Nome do modelo para o título (ex: "ResNet-18").
        save_path (str, optional): Caminho para salvar a figura em PNG.
    """
    cm = confusion_matrix(res['test_labels'], res['test_preds'])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=res['classes'], yticklabels=res['classes'],
        title=f'Matriz de Confusão — {model_name}',
        ylabel='Classe Real', xlabel='Classe Predita')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > threshold else 'black')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figura salva em: {save_path}")
        
    plt.show()