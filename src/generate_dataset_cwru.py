"""
Script para Geração Automática do Dataset de Imagens (Escalogramas) para Treinamento da CNN.
Processa todas as classes do CWRU, divide ARQUIVOS .mat em conjuntos de treino, validação
e teste (evitando Data Leakage), e salva as imagens PNG (224x224) em pastas organizadas.
"""

import sys
from pathlib import Path

# Adicionar a raiz do projeto ao PYTHONPATH para importar src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tqdm import tqdm
from src.config import (
    CLASSES,
    PROCESSED_DATA_DIR
)
from src.dataset_cwru import load_cwru_mat_file, segment_signal, get_cwru_strict_splits
from src.cwt_processor import compute_cwt, save_scalogram_png


def create_directory_structure():
    """Cria as pastas de destino para o dataset processado (train, val, test)."""
    splits = ["train", "val", "test"]
    for split in splits:
        for cls_name in CLASSES:
            path = PROCESSED_DATA_DIR / split / cls_name
            path.mkdir(parents=True, exist_ok=True)


def process_and_generate_dataset(max_windows_per_file: int = None):
    """
    Pipeline de geração do dataset CWRU com Divisão Estrita por Condição de Carga (Leave-One-Load-Out):
      - Treino (20 arquivos): Cargas 0 HP (1797 RPM) e 1 HP (1772 RPM).
      - Validação (10 arquivos): Carga 2 HP (1750 RPM).
      - Teste (10 arquivos): Carga 3 HP (1730 RPM) — condição inédita para o modelo!

    Garante prevenção total de Data Leakage temporal e avalia a capacidade de generalização
    entre regimes operacionais de torque e velocidade distintos.
    """
    print("==========================================================")
    print(" GERAÇÃO DO DATASET DE ESCALOGRAMAS (CWT -> CNN) — CWRU ")
    print("    Divisão Estrita: Leave-One-Load-Out (0HP+1HP -> 2HP -> 3HP) ")
    print("==========================================================")

    create_directory_structure()
    splits = get_cwru_strict_splits()

    total_images_saved = 0

    for cls_name in CLASSES:
        print(f"\n[+] Processando classe: '{cls_name}'...")

        for split_name in ["train", "val", "test"]:
            file_list = splits[split_name][cls_name]
            file_names = [f.stem for f in file_list]
            print(f"    {split_name:>5}: {len(file_list)} arquivos → {file_names}")

            dest_dir = PROCESSED_DATA_DIR / split_name / cls_name
            idx = 0

            for mat_file in file_list:
                signal = load_cwru_mat_file(mat_file)
                windows = segment_signal(signal)
                if max_windows_per_file is not None:
                    windows = windows[:max_windows_per_file]

                for window in tqdm(windows, desc=f"      {cls_name}/{split_name}/{mat_file.stem}", leave=False):
                    scalogram_mat = compute_cwt(window)
                    img_path = dest_dir / f"{cls_name}_{split_name}_{idx:05d}.png"
                    save_scalogram_png(scalogram_mat, str(img_path))
                    idx += 1
                    total_images_saved += 1

            print(f"    [OK] {split_name}: {idx} imagens geradas")

    print("\n==========================================================")
    print(f" Concluído: Total de {total_images_saved} imagens salvas em:")
    print(f" {PROCESSED_DATA_DIR}")
    print("==========================================================")


if __name__ == "__main__":
    process_and_generate_dataset()


