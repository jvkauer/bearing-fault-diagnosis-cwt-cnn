"""
Configuração global de hiperparâmetros, caminhos e constantes do projeto CWRU CWT-CNN.
Projeto de Engenharia de Computação - Diagnóstico de Falhas em Rolamentos.
"""

from pathlib import Path

# ==============================================================================
# 1. Diretórios e Caminhos do Projeto
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Diretórios CWRU
CWRU_DIR = DATA_DIR / "cwru"
RAW_DATA_DIR = CWRU_DIR
PROCESSED_DATA_DIR = CWRU_DIR / "processed"

# Diretórios Paderborn (PU Dataset)
PADERBORN_DIR = DATA_DIR / "paderborn"
PADERBORN_RAW_DIR = PADERBORN_DIR / "raw"
PADERBORN_PROCESSED_DIR = PADERBORN_DIR / "processed"

# Classes do CWRU
CLASSES = ["normal", "inner_race", "outer_race", "ball"]
NUM_CLASSES = len(CLASSES)

# ==============================================================================
# 2. Parâmetros do Sinal e CWRU
# ==============================================================================
# Taxa de Amostragem do acelerômetro Drive End (DE) - 12 kHz
FS = 12_000  # Hz

# Comprimento da janela temporal (em amostras)
WINDOW_SIZE = 1024  # amostras (~85.3 ms)

# Passo do janelamento (Overlap): 512 amostras (50% de sobreposição)
STEP_SIZE = 512  # amostras

# ==============================================================================
# 3. Parâmetros da Transformada Wavelet Contínua (CWT)
# ==============================================================================
# Wavelet de Morlet Complexa (largura de banda B=1.5, freq. central C=1.0)
WAVELET = "cmor1.5-1.0"

# Faixa de frequências de interesse para análise de falhas em rolamentos (Hz)
FREQ_MIN = 10      # Hz (deve ser > 0 para evitar divisão por zero)
FREQ_MAX = 6000    # Hz

# Resolução espacial do escalograma (compatível com CNNs como ResNet, VGG, etc.)
IMG_HEIGHT = 224    # pixels (altura/frequências)
IMG_WIDTH = 224     # pixels (largura/tempo)

# ==============================================================================
# 4. Hiperparâmetros do Modelo e Treinamento da CNN
# ==============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.00005  # 5e-5 (ponto ótimo determinado no estudo de ablação)
NUM_EPOCHS = 10
DROPOUT_RATE = 0.3       # 0.3 (preserva harmônicos espectrais na camada de classificação)

# Seed para reprodutibilidade dos experimentos
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Fixa a semente para garantir reprodutibilidade completa no PyTorch, NumPy e Python.
    """
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 5. Parâmetros do Dataset Paderborn (PU Dataset)
# ==============================================================================
PADERBORN_CLASSES = ["normal", "inner_race", "outer_race"]
PADERBORN_FOLDER_MAP = {
    "K001": "normal",
    "KI14": "inner_race",
    "KA15": "outer_race"
}
PADERBORN_FS = 64_000         # Taxa de amostragem de 64 kHz
PADERBORN_WINDOW_SIZE = 4096  # Comprimento da janela (~64 ms)
PADERBORN_STEP_SIZE = 2048    # Passo do janelamento (50% de overlap)
PADERBORN_FREQ_MIN = 10       # Hz
PADERBORN_FREQ_MAX = 20_000   # Hz (faixa estendida de alta frequência)

# ==============================================================================
# 6. Parâmetros do Computed Order Tracking (Ângulo-Ordem)
# ==============================================================================
ORDER_TRACKING_DIR = DATA_DIR / "order_tracking"
ORDER_CWRU_DIR = ORDER_TRACKING_DIR / "cwru"
ORDER_PADERBORN_DIR = ORDER_TRACKING_DIR / "paderborn"
ORDER_XJTU_DIR = ORDER_TRACKING_DIR / "xjtu"

ORDER_SAMPLES_PER_REV = 1024  # Amostras uniformes por volta do eixo
ORDER_NUM_REVS = 4            # Janela de 4 voltas completas do eixo
ORDER_MIN = 0.5               # Ordem mínima (0.5x da rotação do eixo)
ORDER_MAX = 15.0              # Ordem máxima (15x da rotação do eixo)

# ==============================================================================
# 7. Parâmetros do Dataset XJTU-SY
# ==============================================================================
XJTU_DIR = DATA_DIR / "xjtu"
XJTU_RAW_DIR = XJTU_DIR / "raw"
XJTU_PROCESSED_DIR = XJTU_DIR / "processed"
XJTU_CLASSES = ["normal", "inner_race", "outer_race"]
XJTU_FS = 25_600         # Taxa de amostragem de 25.6 kHz
XJTU_WINDOW_SIZE = 2048  # Comprimento da janela (~80 ms / ~2.8 voltas do eixo)
XJTU_STEP_SIZE = 1024    # Passo do janelamento (50% de overlap)
XJTU_FREQ_MIN = 10       # Hz
XJTU_FREQ_MAX = 10_000   # Hz


