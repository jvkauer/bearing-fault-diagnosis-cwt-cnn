"""
Análise de desempenho computacional dos modelos treinados no CWRU.

Mede, para cada arquitetura (BearingCNN, ResNet-18, EfficientNet-B0, Inception-v3):
    - Número de parâmetros (M);
    - Custo computacional em GFLOPs por imagem (independente de hardware);
    - Tempo médio de treinamento por época em GPU (s);
    - Latência de inferência por amostra (batch = 1) em GPU e em CPU (ms).

Mede também o tempo de pré-processamento (geração do escalograma CWT de uma
janela de 1.024 amostras), permitindo comparar o tempo total de diagnóstico
com a duração da janela de sinal (~85,33 ms a 12 kHz) e verificar a viabilidade
de operação em tempo real.

Fornece ainda as funções `plot_performance_comparison` e `plot_tradeoff_single`
para gerar os gráficos de compromisso (trade-off) entre acurácia, latência,
parâmetros e GFLOPs no padrão da monografia.

Uso:
    python src/benchmark_desempenho.py
"""

import gc
import io
import sys
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import FS, WINDOW_SIZE, BATCH_SIZE
from src.cnn_processor import BearingCNN
from src.transfer_learning import build_transfer_model
from src.cwt_processor import compute_cwt, scalogram_to_rgb

CKPT_DIR = BASE_DIR / "notebooks" / "1_cwru"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "docs" / "images"
NUM_CLASSES = 4

MODELS = {
    "BearingCNN": ("bearing_cnn", CKPT_DIR / "checkpoint_bearing_cnn_best.pth", 97.44),
    "ResNet-18": ("resnet18", CKPT_DIR / "checkpoint_resnet18_best.pth", 99.97),
    "EfficientNet-B0": ("efficientnet_b0", CKPT_DIR / "checkpoint_efficientnet_b0_best.pth", 97.96),
    "Inception-v3": ("inception_v3", CKPT_DIR / "checkpoint_inception_v3_best.pth", 99.25),
}

WARMUP = 20
RUNS_GPU = 200
RUNS_CPU = 50
RUNS_CWT = 50


def load_model(model_key: str, ckpt_path: Path):
    """Constrói a arquitetura, carrega os pesos treinados e retorna (modelo, img_size)."""
    if model_key == "bearing_cnn":
        model, img_size = BearingCNN(num_classes=NUM_CLASSES), 224
    else:
        model, img_size = build_transfer_model(model_key, num_classes=NUM_CLASSES, freeze_backbone=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.eval(), img_size


def count_gflops(model, img_size: int) -> float:
    """Conta as operações de ponto flutuante de uma inferência (batch = 1)."""
    x = torch.randn(1, 3, img_size, img_size)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x)
    return counter.get_total_flops() / 1e9


def measure_latency(model, img_size: int, device: torch.device, runs: int) -> np.ndarray:
    """Mede a latência de inferência (ms) por amostra, com aquecimento e sincronização."""
    model = model.to(device)
    x = torch.randn(1, 3, img_size, img_size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    times = []
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        sync()
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def measure_epoch_training_time(model_key: str, img_size: int, device: torch.device,
                                train_batches: int = 201, val_batches: int = 74) -> float:
    """
    Estima o tempo médio (s) de uma época completa (treino + validação) no CWRU
    (6.416 imagens de treino = 201 batches; 2.339 de validação = 74 batches, batch_size=32).
    """
    if device.type != "cuda":
        return float("nan")

    gc.collect()
    torch.cuda.empty_cache()

    if model_key == "bearing_cnn":
        model = BearingCNN(num_classes=NUM_CLASSES).to(device)
    else:
        model, _ = build_transfer_model(model_key, num_classes=NUM_CLASSES, freeze_backbone=False)
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(BATCH_SIZE, 3, img_size, img_size, device=device)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

    # Warmup
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.logits, y) if hasattr(out, "logits") else criterion(out, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Medição de steps de treino
    n_sample_steps = 30
    t0 = time.perf_counter()
    for _ in range(n_sample_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.logits, y) if hasattr(out, "logits") else criterion(out, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    step_train_s = (time.perf_counter() - t0) / n_sample_steps

    # Medição de steps de validação
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_sample_steps):
            _ = model(x)
    torch.cuda.synchronize()
    step_val_s = (time.perf_counter() - t0) / n_sample_steps

    del model, optimizer, criterion, x, y, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return train_batches * step_train_s + val_batches * step_val_s


def measure_cwt_time() -> np.ndarray:
    """Mede o tempo (ms) para gerar um escalograma RGB 224x224 a partir de uma janela."""
    rng = np.random.default_rng(42)
    window = rng.standard_normal(WINDOW_SIZE)
    for _ in range(3):
        scalogram_to_rgb(compute_cwt(window))

    times = []
    for _ in range(RUNS_CWT):
        t0 = time.perf_counter()
        scalogram_to_rgb(compute_cwt(window))
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def load_benchmark_dataframe(csv_path: Path = RESULTS_DIR / "benchmark_desempenho.csv") -> pd.DataFrame:
    """Carrega o arquivo CSV de benchmark ignorando linhas de comentário."""
    return pd.read_csv(csv_path, comment="#")


def plot_performance_comparison(df: pd.DataFrame = None, cwt_ms: float = 49.47,
                                window_ms: float = 85.33, save_path: Path = None,
                                show: bool = True):
    """
    Gera a figura oficial de compromisso (Trade-off) com 2 painéis relacionando Acurácia
    com Latência de Inferência, Número de Parâmetros (M) e Complexidade Teórica (GFLOPs):
        (a) Acurácia no Teste CWRU (%) vs. Latência de Inferência em GPU (ms) [Bolha = Parâmetros]
        (b) Acurácia no Teste CWRU (%) vs. Ciclo Completo em CPU (CWT + CNN em ms) [Bolha = GFLOPs]
    """
    if df is None:
        df = load_benchmark_dataframe()

    cores = {
        "BearingCNN": "#1f78b4",      # Azul
        "ResNet-18": "#2ca02c",       # Verde
        "EfficientNet-B0": "#ff7f00", # Laranja
        "Inception-v3": "#6a3d9a",    # Roxo
    }

    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)

    # =========================================================================
    # Painel (a): Acurácia (%) vs. Latência na GPU (ms) | Bolha = Parâmetros (M)
    # =========================================================================
    ax_a = axs[0]

    # Fronteira de Pareto (BearingCNN -> ResNet-18)
    pareto_df = df[df["modelo"].isin(["BearingCNN", "ResNet-18"])].sort_values("gpu_ms_media")
    ax_a.plot(
        pareto_df["gpu_ms_media"], pareto_df["acuracia_cwru"],
        linestyle="--", linewidth=2.0, color="#2ca02c", alpha=0.80, zorder=2
    )

    # Posições em pontos (offset points) relativas ao centro de cada bolha
    label_cfg_a = {
        "BearingCNN": ((14, -18), "left", "top"),
        "ResNet-18": ((22, -6), "left", "center"),
        "EfficientNet-B0": ((-18, -16), "right", "top"),
        "Inception-v3": ((-26, -6), "right", "center"),
    }

    for _, row in df.iterrows():
        nome = row["modelo"]
        cor = cores.get(nome, "#333333")
        s_bolha = row["parametros_M"] * 42 + 140

        ax_a.scatter(
            row["gpu_ms_media"], row["acuracia_cwru"], s=s_bolha,
            color=cor, alpha=0.92, edgecolors="black", linewidth=1.6, zorder=4
        )

        (ox, oy), ha_align, va_align = label_cfg_a.get(nome, ((12, 0), "left", "center"))
        ax_a.annotate(
            f"{nome}\n({row['gpu_ms_media']:.2f} ms | {row['acuracia_cwru']:.2f}%)\n[{row['parametros_M']:.2f} M | {row['gflops']:.2f} GFLOPs]",
            xy=(row["gpu_ms_media"], row["acuracia_cwru"]),
            xytext=(ox, oy),
            textcoords="offset points",
            ha=ha_align, va=va_align,
            fontsize=8.8, fontweight="bold", color=cor,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=cor, alpha=0.94, lw=1.1),
            arrowprops=dict(arrowstyle="-", color=cor, lw=1.1, shrinkA=0, shrinkB=6),
            zorder=5
        )

    ax_a.set_title("(a) Acurácia vs. Latência em GPU (Tamanho ∝ Parâmetros)", fontsize=11.5, fontweight="bold", pad=10)
    ax_a.set_xlabel("Latência de Inferência em GPU por Amostra (ms)", fontsize=10.8)
    ax_a.set_ylabel("Acurácia Multiclasse no Teste CWRU (%)", fontsize=10.8)
    ax_a.set_xlim(-0.5, max(df["gpu_ms_media"]) * 1.18)
    ax_a.set_ylim(96.3, 100.45)
    ax_a.grid(True, linestyle="--", alpha=0.5, zorder=1)

    legend_handles_a = [
        Line2D([0], [0], marker="o", color="w", label=f"{r['modelo']} ({r['parametros_M']:.2f} M parâm.)",
               markerfacecolor=cores.get(r["modelo"], "#333"), markeredgecolor="black", markersize=8.5)
        for _, r in df.iterrows()
    ]
    legend_handles_a.append(
        Line2D([0], [0], linestyle="--", color="#2ca02c", lw=2.0, label="Fronteira de Eficiência (Pareto)")
    )
    ax_a.legend(handles=legend_handles_a, title="Arquiteturas (Bolha ∝ Parâmetros)",
                fontsize=8.8, title_fontsize=9.2, loc="lower right", framealpha=0.96)

    # =========================================================================
    # Painel (b): Acurácia (%) vs. Ciclo Completo CPU (CWT + CNN) | Bolha = GFLOPs
    # =========================================================================
    ax_b = axs[1]

    max_x_b = max(df["total_cpu_ms"]) * 1.14
    ax_b.axvspan(20, window_ms, color="#e5f5e0", alpha=0.55, zorder=0)
    ax_b.axvspan(window_ms, max_x_b, color="#fee0d2", alpha=0.40, zorder=0)
    ax_b.axvline(window_ms, color="#cb181d", linestyle="--", linewidth=2.0, zorder=2)

    ax_b.text(
        window_ms - 2.5, 96.45, f"← Viável em Tempo Real\n    (< {window_ms:.2f} ms)",
        ha="right", va="bottom", fontsize=8.8, fontweight="bold", color="#006d2c"
    )
    ax_b.text(
        window_ms + 2.5, 96.45, f"Excede a Janela Física →\n(> {window_ms:.2f} ms)",
        ha="left", va="bottom", fontsize=8.8, fontweight="bold", color="#a50f15"
    )

    label_cfg_b = {
        "BearingCNN": ((10, 22), "right", "bottom"),
        "ResNet-18": ((20, -6), "left", "center"),
        "EfficientNet-B0": ((18, -10), "left", "top"),
        "Inception-v3": ((-26, -8), "right", "center"),
    }

    for _, row in df.iterrows():
        nome = row["modelo"]
        cor = cores.get(nome, "#333333")
        s_bolha_gf = row["gflops"] * 75 + 140

        ax_b.scatter(
            row["total_cpu_ms"], row["acuracia_cwru"], s=s_bolha_gf,
            color=cor, alpha=0.92, edgecolors="black", linewidth=1.6, zorder=4
        )

        (ox, oy), ha_align, va_align = label_cfg_b.get(nome, ((14, 0), "left", "center"))
        ax_b.annotate(
            f"{nome}\n({row['total_cpu_ms']:.2f} ms | {row['acuracia_cwru']:.2f}%)\n[{row['gflops']:.2f} GFLOPs | {row['treino_epoca_s']:.1f} s/época]",
            xy=(row["total_cpu_ms"], row["acuracia_cwru"]),
            xytext=(ox, oy),
            textcoords="offset points",
            ha=ha_align, va=va_align,
            fontsize=8.6, fontweight="bold", color=cor,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=cor, alpha=0.94, lw=1.1),
            arrowprops=dict(arrowstyle="-", color=cor, lw=1.1, shrinkA=0, shrinkB=6),
            zorder=5
        )

    ax_b.set_title(f"(b) Acurácia vs. Ciclo Completo em CPU: CWT ({cwt_ms:.1f} ms) + CNN (Tamanho ∝ GFLOPs)",
                   fontsize=11.5, fontweight="bold", pad=10)
    ax_b.set_xlabel("Tempo Total por Janela em CPU: Extração CWT + Inferência CNN (ms)", fontsize=10.8)
    ax_b.set_xlim(20, max_x_b)
    ax_b.set_ylim(96.3, 100.45)
    ax_b.grid(True, linestyle="--", alpha=0.5, zorder=1)

    legend_handles_b = [
        Line2D([0], [0], marker="o", color="w", label=f"{r['modelo']} ({r['gflops']:.2f} GFLOPs)",
               markerfacecolor=cores.get(r["modelo"], "#333"), markeredgecolor="black", markersize=8.5)
        for _, r in df.iterrows()
    ]
    legend_handles_b.append(
        Line2D([0], [0], linestyle="--", color="#cb181d", lw=2.0, label=f"Limite Tempo Real ({window_ms:.2f} ms)")
    )
    ax_b.legend(handles=legend_handles_b, title="Complexidade Teórica (Bolha ∝ GFLOPs)",
                fontsize=8.8, title_fontsize=9.2, loc="lower right", framealpha=0.96)

    fig.suptitle("Análise de Compromisso (Trade-off): Acurácia vs. Custo Computacional no Conjunto CWRU (3.081 Amostras)",
                 fontsize=13.2, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        save_path.write_bytes(buf.getvalue())
        print(f"[INFO] Figura salva em: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_tradeoff_single(df: pd.DataFrame = None, save_path: Path = None, show: bool = True):
    """
    Gera a versão de painel único do gráfico de compromisso (Trade-off):
    Acurácia no Teste CWRU (%) vs. Latência de Inferência na GPU (ms),
    com tamanho da bolha proporcional aos parâmetros (M) e GFLOPs anotados.
    """
    if df is None:
        df = load_benchmark_dataframe()

    cores = {
        "BearingCNN": "#1f78b4",      # Azul
        "ResNet-18": "#2ca02c",       # Verde
        "EfficientNet-B0": "#ff7f00", # Laranja
        "Inception-v3": "#6a3d9a",    # Roxo
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    pareto_df = df[df["modelo"].isin(["BearingCNN", "ResNet-18"])].sort_values("gpu_ms_media")
    ax.plot(
        pareto_df["gpu_ms_media"], pareto_df["acuracia_cwru"],
        linestyle="--", linewidth=2.0, color="#2ca02c", alpha=0.80, zorder=2
    )

    label_cfg = {
        "BearingCNN": ((14, -16), "left", "top"),
        "ResNet-18": ((22, -6), "left", "center"),
        "EfficientNet-B0": ((-18, -16), "right", "top"),
        "Inception-v3": ((-26, -6), "right", "center"),
    }

    for _, row in df.iterrows():
        nome = row["modelo"]
        cor = cores.get(nome, "#333333")
        tamanho_bolha = row["parametros_M"] * 45 + 140

        ax.scatter(
            row["gpu_ms_media"], row["acuracia_cwru"], s=tamanho_bolha,
            color=cor, alpha=0.92, edgecolors="black", linewidth=1.6, zorder=4
        )
        (ox, oy), ha_align, va_align = label_cfg.get(nome, ((12, 0), "left", "center"))
        ax.annotate(
            f"{nome}\n({row['gpu_ms_media']:.2f} ms | {row['acuracia_cwru']:.2f}%)\n[{row['parametros_M']:.2f} M | {row['gflops']:.2f} GFLOPs]",
            xy=(row["gpu_ms_media"], row["acuracia_cwru"]),
            xytext=(ox, oy),
            textcoords="offset points",
            ha=ha_align, va=va_align,
            fontsize=8.8, fontweight="bold", color=cor,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=cor, alpha=0.94, lw=1.1),
            arrowprops=dict(arrowstyle="-", color=cor, lw=1.1, shrinkA=0, shrinkB=6),
            zorder=5
        )

    ax.set_title("Compromisso (Trade-off): Acurácia no Teste CWRU vs. Latência de Inferência em GPU",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Latência de Inferência em GPU por Amostra (ms)", fontsize=11)
    ax.set_ylabel("Acurácia Multiclasse no Teste CWRU (%)", fontsize=11)
    ax.set_ylim(96.3, 100.45)
    ax.set_xlim(-0.5, max(df["gpu_ms_media"]) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               label=f"{r['modelo']} ({r['parametros_M']:.2f} M parâm. | {r['gflops']:.2f} GFLOPs)",
               markerfacecolor=cores.get(r["modelo"], "#333"), markeredgecolor="black", markersize=8.5)
        for _, r in df.iterrows()
    ]
    legend_handles.append(
        Line2D([0], [0], linestyle="--", color="#2ca02c", lw=2.0, label="Fronteira de Eficiência (Pareto)")
    )
    ax.legend(handles=legend_handles, title="Tamanho da bolha ∝ Parâmetros",
              fontsize=8.8, title_fontsize=9.2, loc="lower right", framealpha=0.96)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        save_path.write_bytes(buf.getvalue())
        print(f"[INFO] Gráfico de trade-off salvo em: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    torch.manual_seed(42)
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"
    window_ms = 1000.0 * WINDOW_SIZE / FS

    print("=" * 92)
    print(" ANÁLISE DE DESEMPENHO COMPUTACIONAL (CWRU)")
    print("=" * 92)
    print(f"GPU: {gpu_name}")
    print(f"CPU: {platform.processor()} | Threads PyTorch: {torch.get_num_threads()}")
    print(f"PyTorch {torch.__version__} | Duração da janela: {window_ms:.2f} ms\n")

    cwt_t = measure_cwt_time()
    print(f"Geração do escalograma CWT (CPU): {cwt_t.mean():.2f} ± {cwt_t.std():.2f} ms\n")

    rows = []
    for name, (key, ckpt, acc_cwru) in MODELS.items():
        model, img_size = load_model(key, ckpt)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        gflops = count_gflops(model, img_size)
        cpu_t = measure_latency(model, img_size, torch.device("cpu"), RUNS_CPU)
        gpu_t = measure_latency(model, img_size, torch.device("cuda"), RUNS_GPU) if has_gpu else np.array([np.nan])
        del model
        gc.collect()
        if has_gpu:
            torch.cuda.empty_cache()

        epoch_s = measure_epoch_training_time(key, img_size, torch.device("cuda")) if has_gpu else np.nan

        rows.append((name, params_m, gflops, epoch_s, epoch_s * 10.0 / 60.0,
                     gpu_t.mean(), gpu_t.std(), cpu_t.mean(), cpu_t.std(), acc_cwru))
        print(f"[OK] {name}")

    header = (f"{'Modelo':<16}|{'Parâm. (M)':>10}|{'GFLOPs':>8}|{'Treino/Época':>13}|"
              f"{'GPU (ms)':>15}|{'CPU (ms)':>16}|{'Total CPU (ms)':>15}")
    print("\n" + header)
    print("-" * len(header))
    lines = [
        "modelo,parametros_M,gflops,treino_epoca_s,treino_10ep_min,"
        "gpu_ms_media,gpu_ms_desvio,cpu_ms_media,cpu_ms_desvio,total_cpu_ms,acuracia_cwru"
    ]
    for name, p, g, ep_s, ep_10m, gm, gs, cm, cs, acc in rows:
        total = cwt_t.mean() + cm
        print(f"{name:<16}|{p:>10.2f}|{g:>8.2f}|{ep_s:>11.2f} s|"
              f"{gm:>8.2f} ± {gs:<4.2f}|{cm:>9.2f} ± {cs:<4.2f}|{total:>15.2f}")
        lines.append(f"{name},{p:.4f},{g:.4f},{ep_s:.2f},{ep_10m:.2f},"
                     f"{gm:.2f},{gs:.2f},{cm:.2f},{cs:.2f},{total:.2f},{acc:.2f}")

    print(f"\nTotal CPU = CWT ({cwt_t.mean():.2f} ms) + inferência em CPU. Janela de sinal = {window_ms:.2f} ms.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "benchmark_desempenho.csv"
    out.write_text(
        "\n".join(lines) + f"\n# cwt_ms_media={cwt_t.mean():.2f},cwt_ms_desvio={cwt_t.std():.2f},"
        f"janela_ms={window_ms:.2f},gpu={gpu_name},cpu={platform.processor()}\n",
        encoding="utf-8"
    )
    print(f"Resultados salvos em: {out}")


if __name__ == "__main__":
    main()
