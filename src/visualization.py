"""
Adicionar este bloco ao final de src/visualization.py.
Substitui o código de plot duplicado nos 4 lugares (cwru_treinamento_cnn.ipynb
e as 3 chamadas em cwru_transfer_learning.ipynb).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_curves(
    historico: dict,
    model_name: str,
    save_path: str = None
):
    """
    Plota as curvas de aprendizado (Perda e Acurácia) de treino/validação,
    destacando a época do checkpoint (menor val_loss) salvo durante o treino.

    Args:
        historico (dict): dicionário com listas 'train_loss', 'val_loss',
                           'train_acc', 'val_acc' (retornado pelas funções
                           de treino em cnn_processor.py / transfer_learning.py).
        model_name (str): nome do modelo, usado nos títulos (ex: "ResNet18").
        save_path (str, optional): caminho para salvar a figura em PNG.
    """
    epocas_eixo = range(1, len(historico["train_loss"]) + 1)

    # Época do checkpoint = menor val_loss (mesmo critério usado no treino)
    best_epoch = int(np.argmin(historico["val_loss"])) + 1
    best_val_loss = min(historico["val_loss"])
    best_val_acc = historico["val_acc"][best_epoch - 1]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --------------------------------------------------------------------
    # Painel (a): Curva de Perda
    # --------------------------------------------------------------------
    axs[0].plot(epocas_eixo, historico["train_loss"],
                label=f"Treino (final: {historico['train_loss'][-1]:.4f})",
                marker="o", color="#3182bd", linewidth=1.8)
    axs[0].plot(epocas_eixo, historico["val_loss"],
                label=f"Validação (final: {historico['val_loss'][-1]:.4f})",
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
    axs[1].plot(epocas_eixo, historico["train_acc"],
                label=f"Treino (final: {historico['train_acc'][-1]:.2f}%)",
                marker="o", color="#3182bd", linewidth=1.8)
    axs[1].plot(epocas_eixo, historico["val_acc"],
                label=f"Validação (final: {historico['val_acc'][-1]:.2f}%)",
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
        print(f"Figura salva com sucesso em: {save_path}")

    plt.show()