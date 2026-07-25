"""
Configuração global de hiperparâmetros e caminhos do projeto CWRU CWT-CNN.
Projeto de Engenharia de Computação - Diagnóstico de Falhas em Rolamentos.
"""

from pathlib import Path

# ==============================================================================
# 1. Diretores e Caminhos
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Subpastas de dados brutos (.mat)
CLASSES = ["normal", "inner_race", "outer_race", "ball"]

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
FREQ_MIN = 50      # Hz
FREQ_MAX = 4000    # Hz

# Resolução vertical do escalograma (compatível com CNNs como ResNet, VGG, etc.)
IMG_HEIGHT = 224    # pixels
IMG_WIDTH = 224     # pixels

# Seed para reprodutibilidade dos experimentos
RANDOM_SEED = 42
