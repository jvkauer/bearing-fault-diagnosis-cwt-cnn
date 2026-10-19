"""
Análise de desempenho computacional dos modelos treinados no CWRU.

Mede, para cada arquitetura (BearingCNN, ResNet-18, EfficientNet-B0, Inception-v3):
    - Número de parâmetros (M);
    - Custo computacional em GFLOPs por imagem (independente de hardware);
    - Tempo médio de treinamento por época em GPU (s);
    - Latência de inferência por amostra (batch = 1) em GPU e em CPU (ms).

Mede também o tempo de pré-processamento (geração do escalograma CWT de uma
janela de 1.024 amostras), permitindo comparar o tempo total de diagnóstico
com a duração da janela de sinal (~85,3 ms a 12 kHz) e verificar a viabilidade
de operação em tempo real.

Fornece ainda a função `plot_performance_comparison` para gerar os gráficos
comparativos de desempenho computacional no padrão da monografia.

Uso:
    python src/benchmark_desempenho.py
"""

import sys
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                                train_batches: int = 222, val_batches: int = 74) -> float:
    """
    Estima o tempo médio (s) de uma época completa (treino + validação) no CWRU
    (7.100 imagens de treino = 222 batches; 2.364 de validação = 74 batches, batch_size=32).
    """
    if device.type != "cuda":
        return float("nan")

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

    del model, x, y
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


def plot_performance_comparison(df: pd.DataFrame = None, cwt_ms: float = 64.06,
                                window_ms: float = 85.33, save_path: Path = None):
    """
    Gera uma figura com 3 painéis comparando o desempenho computacional das 4 arquiteturas:
        (a) Latência de Inferência por Amostra (GPU vs. CPU)
        (b) Viabilidade em Tempo Real na CPU (Pré-processamento CWT + Inferência CNN vs. Janela de 85,3 ms)
        (c) Tempo de Treinamento por Época vs. Número de Parâmetros e GFLOPs
    """
    if df is None:
        df = load_benchmark_dataframe()

    modelos = df["modelo"].tolist()
    x = np.arange(len(modelos))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    # -------------------------------------------------------------------------
    # Painel (a): Latência de Inferência (GPU vs CPU)
    # -------------------------------------------------------------------------
    width = 0.36
    bars_gpu = axs[0].bar(x - width / 2, df["gpu_ms_media"], width, yerr=df["gpu_ms_desvio"],
                          capsize=4, label="GPU (RTX 2060 SUPER)", color="#2b8cbe", edgecolor="black", alpha=0.9)
    bars_cpu = axs[0].bar(x + width / 2, df["cpu_ms_media"], width, yerr=df["cpu_ms_desvio"],
                          capsize=4, label="CPU (Ryzen 5 2600)", color="#e34a33", edgecolor="black", alpha=0.9)

    axs[0].set_title("(a) Latência de Inferência por Amostra", fontsize=12, fontweight="bold")
    axs[0].set_ylabel("Tempo por Amostra (ms)", fontsize=11)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(modelos, rotation=15, ha="right", fontsize=10)
    axs[0].set_ylim(0, max(df["cpu_ms_media"] + df["cpu_ms_desvio"]) * 1.18)
    axs[0].grid(axis="y", linestyle="--", alpha=0.5)
    axs[0].legend(fontsize=9.5, loc="upper left")

    for bar, std in zip(bars_gpu, df["gpu_ms_desvio"]):
        h = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, h + std + 2.5, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#045a8d")
    for bar, std in zip(bars_cpu, df["cpu_ms_desvio"]):
        h = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, h + std + 2.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#b30000")

    # -------------------------------------------------------------------------
    # Painel (b): Viabilidade em Tempo Real na CPU (CWT + CNN vs Janela 85,3 ms)
    # -------------------------------------------------------------------------
    bars_cwt = axs[1].bar(x, [cwt_ms] * len(modelos), width=0.5,
                          label=f"Extração CWT ({cwt_ms:.1f} ms)", color="#74c476", edgecolor="black", alpha=0.9)
    bars_cnn = axs[1].bar(x, df["cpu_ms_media"], width=0.5, bottom=[cwt_ms] * len(modelos),
                          label="Inferência CNN na CPU", color="#fd8d3c", edgecolor="black", alpha=0.9)
    axs[1].axhline(window_ms, color="#d90000", linestyle="--", linewidth=2.0,
                   label=f"Limite Tempo Real ({window_ms:.1f} ms)")

    axs[1].set_title("(b) Ciclo Completo em CPU (CWT + CNN)", fontsize=12, fontweight="bold")
    axs[1].set_ylabel("Tempo Total por Janela (ms)", fontsize=11)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(modelos, rotation=15, ha="right", fontsize=10)
    axs[1].set_ylim(0, max(df["total_cpu_ms"]) * 1.18)
    axs[1].grid(axis="y", linestyle="--", alpha=0.5)
    axs[1].legend(fontsize=9.5, loc="upper left")

    for i, total in enumerate(df["total_cpu_ms"]):
        status_color = "#006d2c" if total <= window_ms else "#b30000"
        axs[1].text(x[i], total + 3.5, f"{total:.2f} ms",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=status_color)

    # -------------------------------------------------------------------------
    # Painel (c): Tempo de Treinamento por Época (GPU) e GFLOPs
    # -------------------------------------------------------------------------
    bars_train = axs[2].bar(x, df["treino_epoca_s"], width=0.5,
                            color="#756bb1", edgecolor="black", alpha=0.9,
                            label="Treino + Validação por Época (GPU)")
    axs[2].set_title("(c) Tempo de Treinamento por Época (GPU)", fontsize=12, fontweight="bold")
    axs[2].set_ylabel("Tempo por Época (s)", fontsize=11)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(modelos, rotation=15, ha="right", fontsize=10)
    axs[2].set_xlim(-0.55, len(modelos) - 0.35)
    axs[2].set_ylim(0, max(df["treino_epoca_s"]) * 1.24)
    axs[2].grid(axis="y", linestyle="--", alpha=0.5)
    axs[2].legend(fontsize=9.5, loc="upper left")

    for i, bar in enumerate(bars_train):
        h = bar.get_height()
        p_m = df["parametros_M"].iloc[i]
        gf = df["gflops"].iloc[i]
        axs[2].text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                    f"{h:.2f} s\n({p_m:.2f}M | {gf:.2f} GF)",
                    ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#3f007d")

    fig.suptitle("Análise Comparativa de Desempenho Computacional no Conjunto CWRU",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figura salva em: {save_path}")

    plt.show()


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
    print(f"PyTorch {torch.__version__} | Duração da janela: {window_ms:.1f} ms\n")

    cwt_t = measure_cwt_time()
    print(f"Geração do escalograma CWT (CPU): {cwt_t.mean():.2f} ± {cwt_t.std():.2f} ms\n")

    rows = []
    for name, (key, ckpt, acc_cwru) in MODELS.items():
        model, img_size = load_model(key, ckpt)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        gflops = count_gflops(model, img_size)
        cpu_t = measure_latency(model, img_size, torch.device("cpu"), RUNS_CPU)
        gpu_t = measure_latency(model, img_size, torch.device("cuda"), RUNS_GPU) if has_gpu else np.array([np.nan])
        epoch_s = measure_epoch_training_time(key, img_size, torch.device("cuda")) if has_gpu else np.nan

        rows.append((name, params_m, gflops, epoch_s, epoch_s * 10.0 / 60.0,
                     gpu_t.mean(), gpu_t.std(), cpu_t.mean(), cpu_t.std(), acc_cwru))
        print(f"[OK] {name}")
        del model
        if has_gpu:
            torch.cuda.empty_cache()

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

    print(f"\nTotal CPU = CWT ({cwt_t.mean():.2f} ms) + inferência em CPU. Janela de sinal = {window_ms:.1f} ms.")

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
