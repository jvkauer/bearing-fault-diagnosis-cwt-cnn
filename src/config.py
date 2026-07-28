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
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"

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
LEARNING_RATE = 0.0005
NUM_EPOCHS = 15
DROPOUT_RATE = 0.5

# Seed para reprodutibilidade dos experimentos
RANDOM_SEED = 42
