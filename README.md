# Diagnóstico de Falhas em Rolamentos com CWT, Computed Order Tracking e Deep Learning

Transformada Wavelet Contínua (CWT), *Computed Order Tracking* (COT) e Redes Neurais Convolucionais aplicadas ao diagnóstico precoce de falhas em mancais de rolamento sob variações de velocidade, geometria e severidade operacional.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/Datasets-CWRU%20%7C%20Paderborn%20%7C%20XJTU--SY-blue)](https://engineering.case.edu/bearingdatacenter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Sumário

- [Visão Geral](#visão-geral)
- [Bancos de Dados Analisados (Benchmark Tripartite)](#bancos-de-dados-analisados-benchmark-tripartite)
- [Justificativa Técnica: CWT vs. STFT vs. FFT](#justificativa-técnica-cwt-vs-stft-vs-fft)
- [Computed Order Tracking (COT) e Invariância à Rotação](#computed-order-tracking-cot-e-invariância-à-rotação)
- [Escalogramas por Classe](#escalogramas-por-classe)
- [Pipeline do Sistema](#pipeline-do-sistema)
- [Resultados Intra-Domínio (In-Domain)](#resultados-intra-domínio-in-domain)
- [Avaliação Cruzada e Análise de Domain Shift (Cross-Domain)](#avaliação-cruzada-e-análise-de-domain-shift-cross-domain)
- [Adaptação de Domínio com Poucas Amostras (Few-Shot Domain Adaptation)](#adaptação-de-domínio-com-poucas-amostras-few-shot-domain-adaptation)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Guia de Uso](#guia-de-uso)
- [Modelos de Deep Learning e Transfer Learning](#modelos-de-deep-learning-e-transfer-learning)
- [Base Teórica e Referências](#base-teórica-e-referências)
- [Roadmap do Projeto](#roadmap-do-projeto)
- [Autor e Licença](#autor-e-licença)

---

## Visão Geral

Mancais de rolamento respondem por mais de 40% das falhas catastróficas em máquinas elétricas rotativas industriais (motores de indução, geradores, bombas e compressores). O diagnóstico precoce dessas anomalias é um pilar indispensável para a manutenção preditiva da Indústria 4.0.

Este projeto desenvolve e valida experimentalmente uma metodologia robusta de diagnóstico que combina:

1. **Transformada Wavelet Contínua (CWT):** Mapeia os sinais temporais unidimensionais de vibração em escalogramas bidimensionais tempo-frequência (wavelet Morlet complexa), capturando com alta resolução eventos transitórios não estacionários provocados pelo choque dos elementos rolantes sobre descontinuidades mecânicas.
2. **Computed Order Tracking (COT):** Reamostra o sinal temporal para o domínio angular (amostras uniformes por revolução do eixo), gerando escalogramas no domínio de **Ângulo-Ordem (Order-CWT)**, eliminando a dependência do sinal em relação à velocidade de rotação (RPM).
3. **Deep Learning e Transfer Learning:** Classificação automática dos escalogramas via CNN customizada (*BearingCNN*, ~1.2M parâmetros) e modelos convolucionais pré-treinados no ImageNet (*ResNet-18*, *EfficientNet-B0*, *Inception-v3*).
4. **Estudo de Generalização Cruzada (*Domain Shift*):** Avaliação tripartite entre falhas artificiais usinadas em laboratório (**CWRU**) e falhas autênticas por fadiga mecânica acelerada (**Paderborn University** e **XJTU-SY**), comprovando a eficácia de *Few-Shot Domain Adaptation* para calibração industrial instantânea.

---

## Bancos de Dados Analisados (Benchmark Tripartite)

O projeto contempla três dos maiores e mais respeitados bancos de dados públicos da literatura internacional de prognóstico e saúde de máquinas (PHM):

| Característica | CWRU (EUA) | Paderborn University (Alemanha) | XJTU-SY (China) |
|---|:---:|:---:|:---:|
| **Rolamento Testado** | SKF 6205-2RS JEM | SKF 6203 | LDK UER204 |
| **Número de Esferas ($Z$)** | 9 esferas | **8 esferas** | **8 esferas** |
| **Ordem Pista Interna (BPFI)** | **5.41 ordens** | **4.95 ordens** | **4.92 ordens** ($\Delta = 0{,}03$) |
| **Ordem Pista Externa (BPFO)** | **3.58 ordens** | **3.05 ordens** | **3.08 ordens** ($\Delta = 0{,}03$) |
| **Taxa de Amostragem ($f_s$)** | 12.000 Hz | 64.000 Hz | 25.600 Hz |
| **Velocidade de Rotação** | ~1.797 RPM (variável) | 900 e 1.500 RPM | 2.100 e 2.250 RPM |
| **Natureza da Falha** | Artificial (eletroerosão EDM) | **Fadiga mecânica real acelerada** | **Fadiga real (*run-to-failure*)** |
| **Condições Avaliadas** | Normal, Pista Interna, Pista Externa, Esfera | Normal, Pista Interna, Pista Externa | Normal, Pista Interna, Pista Externa |

> [!TIP]
> **Compatibilidade Cinemática Paderborn $\leftrightarrow$ XJTU-SY:**  
> Ambos os rolamentos possuem exatamente 8 esferas e relações dimensionais que resultam em ordens de falha virtualmente idênticas ($\text{BPFI} \approx 4{,}9$ e $\text{BPFO} \approx 3{,}0$). Essa simetria geométrica é o alicerce ideal para validação de algoritmos de transferência de domínio entre máquinas reais.

---

## Justificativa Técnica: CWT vs. STFT vs. FFT

| Característica | FFT (Fourier Rápida) | STFT (Fourier de Janela Curta) | CWT (Wavelet Contínua) |
|---|:---:|:---:|:---:|
| **Domínio de Análise** | Frequência pura | Tempo-frequência | Tempo-frequência |
| **Janela de Ponderação** | Sinal inteiro (infinita) | Fixa no tempo e frequência | **Adaptativa multi-resolução** |
| **Resolução Temporal** | Nula | Constante | **Alta para frequências elevadas** |
| **Resolução Espectral** | Constante | Constante | **Alta para baixas frequências** |
| **Sinais Não Estacionários** | Inadequada | Parcialmente adequada | **Excelente (padrão-ouro)** |

Sinais de vibração decorrentes de falhas em rolamentos são inerentemente **não estacionários e impulsivos**. A CWT contorna o Princípio da Incerteza de Heisenberg-Gabor adaptando sua janela de análise através de operações de dilatação (*escala*) e translação (*tempo*), permitindo localizar no tempo pulsos transitórios de alta frequência e capturar harmônicos contínuos em baixa frequência.

<p align="center">
  <img src="docs/images/signal_vs_scalogram.png" width="750" alt="Sinal de vibração bruto e respectivo escalograma CWT">
</p>
<p align="center"><sub>(a) Sinal de vibração bruto no domínio do tempo e (b) escalograma CWT correspondente com wavelet-mãe Morlet complexa (cmor1.5-1.0).</sub></p>

---

## Computed Order Tracking (COT) e Invariância à Rotação

Em aplicações industriais reais, a rotação do motor flutua devido a oscilações de carga e rede. No domínio do tempo, uma mudança de rotação comprime ou expande os pulsos, provocando dispersão espectral e falha catastrófica de modelos treinados em rotação fixa (*Domain Shift*).

O **Computed Order Tracking (COT)** reamostra o sinal temporal $x(t)$ para o domínio angular $x(\theta)$ através de interpolação cúbica baseada no vetor de rotação instantânea do eixo:
$$\Delta \theta = \frac{2\pi}{N_{\text{amostras/volta}}}$$

Ao aplicar a CWT no domínio angular, o eixo vertical passa de frequências em Hertz ($f$) para **Ordens de Rotação** ($O = f / f_r$). As frequências de defeito mecânico (BPFI, BPFO, BSF) tornam-se constantes puramente geométricas:

<p align="center">
  <img src="docs/images/order_tracking_comparison_time_vs_order.png" width="850" alt="Comparação Tempo-Frequência vs Ângulo-Ordem">
</p>
<p align="center"><sub>Comparativo: no domínio do tempo (linha 1), a falha varia de 74 Hz a 160 Hz devido ao RPM. No domínio de Ordens (linha 2), ambos os sinais alinham-se na ordem geométrica correspondente (~5.0 ordens).</sub></p>

---

## Escalogramas por Classe

Cada topologia de falha gera padrões visuais e assinaturas de textura singulares no escalograma bidimensional:

<p align="center">
  <img src="docs/images/scalogram_normal.png" width="180" alt="Normal">
  <img src="docs/images/scalogram_inner_race.png" width="180" alt="Pista Interna">
  <img src="docs/images/scalogram_outer_race.png" width="180" alt="Pista Externa">
  <img src="docs/images/scalogram_ball.png" width="180" alt="Esfera">
</p>
<p align="center"><sub>Da esquerda para a direita: Condição Normal (baixa densidade espectral), Falha na Pista Interna (impactos periódicos de alta frequência modulados pela rotação do eixo), Falha na Pista Externa (pulsos uniformes de alta energia) e Falha na Esfera (modulação dupla por rotação de gaiola e giro da esfera).</sub></p>

---

## Pipeline do Sistema

```mermaid
flowchart TD
    subgraph Entrada["1. Aquisição de Sinais Brutos"]
        A1["CWRU Dataset<br/>(12 kHz / EDM Artificial)"]
        A2["Paderborn University<br/>(64 kHz / Fadiga Real)"]
        A3["XJTU-SY Dataset<br/>(25.6 kHz / Run-to-Failure)"]
    end

    subgraph Processamento["2. Processamento e Representação 2D"]
        B1["Janelamento Temporal Fixo<br/>(1024 a 4096 amostras)"]
        B2["Computed Order Tracking (COT)<br/>(1024 amostras/rev, 4 voltas)"]
        C1["CWT Tempo-Frequência<br/>(cmor1.5-1.0 | 10 Hz a Nyquist)"]
        C2["Order-CWT Ângulo-Ordem<br/>(cmor1.5-1.0 | 0.5 a 15.0 ordens)"]
        D["Escalogramas 224x224 RGB<br/>(Normalização por Percentil 1-99)"]
    end

    subgraph Modelos["3. Aprendizado Profundo"]
        E1["BearingCNN Customizada<br/>(~1.2M parâmetros, treino do zero)"]
        E2["Transfer Learning (Fine-Tuning)<br/>(ResNet18, EfficientNet-B0, Inception-v3)"]
    end

    subgraph Avaliacao["4. Avaliação e Adaptação"]
        F1["Avaliação Intra-Domínio<br/>(Acurácia > 99%)"]
        F2["Avaliação Cruzada (Zero-Shot)<br/>(Evidenciação do Domain Shift)"]
        F3["Few-Shot Domain Adaptation<br/>(Calibração rápida com k=1 a k=5 amostras)"]
    end

    Entrada --> B1 & B2
    B1 --> C1
    B2 --> C2
    C1 & C2 --> D
    D --> E1 & E2
    E1 & E2 --> F1 & F2 & F3
```

---

## Resultados Intra-Domínio (In-Domain)

Resultados obtidos em amostras inéditas de teste utilizando **particionamento estrito baseado em arquivos `.mat` e `.csv` originais** (eliminando qualquer possibilidade de vazamento de dados decorrente de sobreposição de janelas):

### 1. Benchmark CWRU (4 Classes: Normal, Pista Interna, Pista Externa, Esfera)
*2.368 amostras de teste a 12 kHz, $1.797\text{ RPM}$:*

| Modelo | Estratégia | Acurácia Multiclasse | Detecção Binária (Normal vs. Falha) | Parâmetros |
|---|---|:---:|:---:|:---:|
| **ResNet18** | Fine-Tuning (`Freeze=False`) | **99.83%** | **100.00%** | ~11.1M |
| **EfficientNet-B0** | Fine-Tuning (`Freeze=False`) | **98.48%** | **100.00%** | ~5.3M |
| **Inception-v3** | Fine-Tuning (`Freeze=False`) | **97.93%** | **100.00%** | ~23.8M |
| **BearingCNN (Própria)** | Treino do Zero | **97.59%** | **100.00%** | ~1.2M |

<p align="center">
  <img src="docs/images/training_curves_resnet.png" width="580" alt="Curvas de Treino ResNet-18">
  <img src="docs/images/confusion_matrix_resnet.png" width="370" alt="Matriz de Confusão ResNet-18">
</p>
<p align="center"><sub>Curvas de aprendizado com checkpointing automático e Matriz de Confusão consolidada da ResNet-18 no conjunto de teste CWRU (99.83% de acurácia com apenas 4 erros em 2.368 predições).</sub></p>

### 2. Benchmark Paderborn University (3 Classes: Falhas Reais a 64 kHz)
*Amostras de teste com fadiga real por estresse mecânico acelerado:*
- **BearingCNN:** **100.00%** de acurácia no teste (`val_loss: 0.0024`).
- **ResNet-18:** **100.00%** de acurácia no teste.

### 3. Benchmark XJTU-SY (3 Classes: Degradação Acelerada a 25.6 kHz)
*Amostras de teste obtidas em ensaios contínuos de run-to-failure:*
- **BearingCNN:** **100.00%** de acurácia no teste.
- **ResNet-18:** **100.00%** de acurácia no teste.
- **EfficientNet-B0:** **100.00%** de acurácia no teste.

---

## Avaliação Cruzada e Análise de Domain Shift (Cross-Domain)

Apesar dos resultados quase perfeitos obtidos em cada bancada individualmente, a transferência direta (*Zero-Shot Cross-Domain*) de um modelo treinado em um banco de dados para outro sofre com severa degradação:

| Cenário de Teste (Zero-Shot) | Domínio | Acurácia Cruzada | Modo de Falha / Comportamento Observado |
|---|:---:|:---:|---|
| **CWRU $\rightarrow$ Paderborn** | Tempo-Frequência | **10.62%** | Colapso de modo: classificador prevê `ball` em virtude da alta frequência do ruído a 64 kHz |
| **Paderborn $\rightarrow$ CWRU** | Tempo-Frequência | **25.00%** | Colapso de modo: modelo treinado a 900 RPM enxerga CWRU (1797 RPM) com frequências dobradas |
| **CWRU $\rightarrow$ Paderborn** | Ângulo-Ordem (COT) | **33.33%** | Alinhamento do RPM, porém limitado pela disparidade geométrica (BPFI 5.41 vs 4.95 ordens) |
| **Paderborn $\leftrightarrow$ XJTU-SY** | Ângulo-Ordem (COT) | **~50% – 70%** | Alinhamento geométrico perfeito (8 esferas), com interferência da função de resposta estrutural das carcaças |

<p align="center">
  <img src="docs/images/order_tracking_cross_domain_comparison.png" width="700" alt="Comparativo Cross-Domain">
</p>

### Principais Fatores Físicos do Domain Shift:
1. **Diferença de Velocidade (RPM):** O impacto ocorre em intervalos temporais distintos. O *Order Tracking* elimina integralmente este fator.
2. **Diferença Geométrica:** Rolamentos com diâmetros e contagens de esferas distintas possuem ordens fundamentais diferentes.
3. **Assinatura Espectral Real vs. Artificial:** Falhas artificiais (EDM/laser) geram pulsos Dirac impulsivos secos; falhas reais de fadiga produzem espalhamento acústico contínuo por atrito de contato e micro-lascamento.

---

## Adaptação de Domínio com Poucas Amostras (Few-Shot Domain Adaptation)

Para contornar o custo e a inviabilidade prática de rotular milhares de horas de vibração em novos equipamentos industriais, desenvolveu-se uma estratégia de **Few-Shot Domain Adaptation**:
O modelo pré-treinado na bancada de origem (ex: Paderborn ou CWRU) recebe apenas **$k$ amostras rotuladas por classe** ($k=1$ a $k=5$) da máquina-alvo para ajuste fino leve (2 a 3 épocas) do classificador linear final.

<p align="center">
  <img src="docs/images/few_shot_tripartite_full_comparison.png" width="850" alt="Few-Shot Domain Adaptation">
</p>
<p align="center"><sub>Evolução da acurácia cruzada tripartite em função do número de exemplos ($k$-shot). Com apenas $k=5$ amostras calibradoras por classe, a acurácia salta de valores baixos (<35%) para o patamar de 96% a 99.8%.</sub></p>

<p align="center">
  <img src="docs/images/one_shot_k1_time_vs_order_comparison.png" width="750" alt="Comparação 1-Shot">
</p>
<p align="center"><sub>Desempenho no regime extremo de 1-Shot ($k=1$, apenas uma única imagem de calibração por classe): o domínio de Ordens (Order-CWT) acelera e estabiliza a adaptação em comparação ao tempo-frequência clássico.</sub></p>

---

## Estrutura do Repositório

```
bearing-fault-diagnosis-cwt-cnn/
├── data/
│   ├── cwru/                       # Dataset CWRU bruto (.mat) e processado
│   ├── paderborn/                  # Dataset Paderborn University bruto (.mat) e processado
│   ├── xjtu_sy/                    # Dataset XJTU-SY bruto (.csv) e processado
│   └── order_tracking/             # Escalogramas reamostrados em Ângulo-Ordem (Order-CWT)
├── notebooks/
│   ├── 1_cwru/                     # Módulo 1: Pipeline CWRU
│   │   ├── 01_cwru_exploracao_cwt.ipynb
│   │   ├── 02_cwru_treinamento_cnn.ipynb
│   │   └── 03_cwru_transfer_learning.ipynb
│   ├── 2_paderborn/                # Módulo 2: Pipeline Paderborn (64 kHz)
│   │   ├── 01_paderborn_exploracao_cwt.ipynb
│   │   ├── 02_paderborn_treinamento_cnn.ipynb
│   │   └── 03_paderborn_transfer_learning.ipynb
│   ├── 3_cross_domain/             # Módulo 3: Avaliação Cruzada no Tempo
│   │   ├── 01_avaliacao_cruzada_tempo.ipynb
│   │   └── 02_few_shot_tempo_freq.ipynb
│   ├── 4_order_tracking/           # Módulo 4: Computed Order Tracking e Invariância
│   │   ├── 01_exploracao_angulo_ordem.ipynb
│   │   ├── 02_geracao_e_treinamento_order_cwt.ipynb
│   │   ├── 03_avaliacao_cruzada_order.ipynb
│   │   └── 04_few_shot_domain_adaptation.ipynb
│   └── 5_xjtu/                     # Módulo 5: Pipeline XJTU-SY (Run-to-Failure)
│       ├── 01_xjtu_exploracao_cwt.ipynb
│       ├── 02_xjtu_treinamento_cnn.ipynb
│       └── 03_xjtu_transfer_learning.ipynb
├── src/
│   ├── config.py                   # Parâmetros globais, diretórios, geometrias e frequências
│   ├── cwt_processor.py            # Cálculo da CWT e exportação de escalogramas 224x224
│   ├── order_tracking.py           # Algoritmos de Computed Order Tracking (COT) e Order-CWT
│   ├── dataset_cwru.py             # Parser e janelamento dos arquivos .mat do CWRU
│   ├── generate_dataset_cwru.py    # Gerador de escalogramas CWRU sem vazamento de dados
│   ├── dataset_paderborn.py        # Parser e janelamento dos arquivos .mat de Paderborn
│   ├── generate_dataset_paderborn.py # Gerador de escalogramas Paderborn sem data leakage
│   ├── dataset_xjtu.py             # Parser e janelamento dos arquivos .csv do XJTU-SY
│   ├── generate_dataset_xjtu.py    # Gerador de escalogramas XJTU-SY sem data leakage
│   ├── generate_order_dataset.py   # Gerador de escalogramas em Ângulo-Ordem (COT)
│   ├── cnn_processor.py            # Arquitetura BearingCNN e motor de treino/avaliação
│   ├── transfer_learning.py        # Construtores e rotinas de treino para ResNet/Inception/EfficientNet
│   └── visualization.py            # Plotagem de matrizes de confusão, curvas de treino e escalogramas
├── docs/
│   ├── TCC.pdf                     # Monografia completa do trabalho
│   └── images/                     # Figuras de alta resolução utilizadas na documentação
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Instalação

Clone o repositório e configure o ambiente virtual Python:

```bash
git clone https://github.com/jvkauer/bearing-fault-diagnosis-cwt-cnn.git
cd bearing-fault-diagnosis-cwt-cnn

# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente virtual
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

---

## Guia de Uso

### 1. Geração de Escalogramas 2D

Para processar os sinais brutos e gerar os conjuntos de treino, validação e teste com divisão estrita por arquivos:

```bash
# Gerar dataset CWRU (Tempo-Frequência)
python src/generate_dataset_cwru.py

# Gerar dataset Paderborn (Tempo-Frequência)
python src/generate_dataset_paderborn.py

# Gerar dataset XJTU-SY (Tempo-Frequência)
python src/generate_dataset_xjtu.py

# Gerar datasets no domínio de Ângulo-Ordem (Order-CWT)
python src/generate_order_dataset.py
```

### 2. Treinamento de Modelos via Linha de Comando ou Python

```python
from src.transfer_learning import train_and_evaluate_transfer_model

# Treinamento da ResNet-18 com Fine-Tuning
resultado = train_and_evaluate_transfer_model(
    model_name="resnet18",   # Opções: 'resnet18' | 'inception_v3' | 'efficientnet_b0'
    num_epochs=10,
    lr=1e-4,
    freeze_backbone=False,
    data_dir="data/cwru/processed",
    checkpoint_prefix="checkpoint_cwru"
)

print(f"Acurácia no conjunto de teste: {resultado['test_acc']:.2f}%")
```

### 3. Execução Interativa via Jupyter Notebooks
Para reproduzir visualmente cada etapa (curvas de convergência, matrizes de confusão e mapas de ativação), abra os notebooks numerados nas pastas correspondentes em `notebooks/`.

---

## Modelos de Deep Learning e Transfer Learning

| Arquitetura | Profundidade | Parâmetros | Mecanismo Central e Justificativa |
|---|:---:|:---:|---|
| **BearingCNN** | 4 blocos conv | ~1.2M | Rede customizada projetada especificamente para o problema, leve e adequada para *Edge AI*. |
| **ResNet-18** | 18 camadas | ~11.1M | **Modelo Campeão:** Conexões residuais (*skip connections*) que preservam linhas finas de ordem e harmônicos transitórios. |
| **EfficientNet-B0** | Composta | ~5.3M | Escalonamento composto balanceado entre profundidade, largura e resolução com baixa demanda de memória. |
| **Inception-v3** | 42 camadas | ~23.8M | Convoluções paralelas multiescala que capturam assinaturas de impacto em janelas de tempo de diferentes tamanhos. |

---

## Base Teórica e Referências

O desenvolvimento deste trabalho apoia-se nos seguintes referenciais teóricos e metodológicos:

1. **Boudiaf, A. et al. (2016):** *Bearing fault diagnosis using continuous wavelet transform and deep neural networks.*
2. **Guo, X. et al. (2018):** *Deep convolutional transfer learning network: A new method for intelligent fault diagnosis of machines with unseen behaviors.* IEEE TIE.
3. **Kaya, Y., Kuncan, M., & Ertunç, H. M. (2022):** *Bearing fault diagnosis using continuous wavelet transform and CNN transfer learning.* Applied Soft Computing.
4. **Lessmeier, C. et al. (2016):** *Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification.* Paderborn University.
5. **Wang, B. et al. (2019):** *A hybrid prognostics approach for estimating remaining useful life of rolling element bearings.* (Dataset XJTU-SY). IEEE TIE.
6. **He, K. et al. (2016):** *Deep Residual Learning for Image Recognition.* CVPR (ResNet).

---

## Roadmap do Projeto

- [x] Fundamentação teórica de sinais não estacionários (CWT vs. STFT vs. FFT)
- [x] Pipeline de pré-processamento e geração de escalogramas 2D com particionamento estrito sem *Data Leakage*
- [x] Desenvolvimento da arquitetura própria `BearingCNN` (**97.59%** CWRU, **100%** Paderborn, **100%** XJTU-SY)
- [x] Benchmark de *Transfer Learning* com ResNet-18, EfficientNet-B0 e Inception-v3 (**99.83%** CWRU)
- [x] Diagnóstico binário de integridade com **100.00%** de precisão e recall (zero falsos alarmes e zero falsos negativos)
- [x] Integração de bancos de dados com falhas reais de fadiga acelerada (Paderborn University e XJTU-SY)
- [x] Estudo experimental e caracterização da barreira de *Domain Shift* em avaliação cruzada zero-shot
- [x] Implementação de *Computed Order Tracking* (COT) e Transformada Wavelet em Ângulo-Ordem (Order-CWT)
- [x] Metodologia de *Few-Shot Domain Adaptation* demonstrando recuperação da acurácia (>98%) com calibração rápida ($k=1$ a $k=5$)
- [x] Documentação técnica completa e sincronização do repositório acadêmico

---

## Autor e Licença

**João Vitor Kauer Schuck**  
Engenharia de Computação — Universidade Federal de Pelotas (UFPel)  
GitHub: [@jvkauer](https://github.com/jvkauer)

Este projeto está licenciado sob os termos da licença [MIT](LICENSE).
