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
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
DROPOUT_RATE = 0.5

# Seed para reprodutibilidade dos experimentos
RANDOM_SEED = 42

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

