"""
Módulo de Filtros de Aguçamento Espacial (Sharpening) para Escalogramas CWT.
Projeto de Diagnóstico de Falhas em Mancais de Rolamento.
"""

from typing import Tuple, Optional, Callable
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import numpy as np

from src.config import IMG_HEIGHT, IMG_WIDTH


class SharpenTransform:
    """
    Transformação PIL para aguçamento espacial de escalogramas bidimensionais.
    
    Métodos suportados:
    - 'unsharp': Unsharp Masking clássico I_sharp = I + alpha * (I - I_blur)
                 com threshold de ruído (noise gate).
    - 'laplacian': Kernel de derivada de 2ª ordem 3x3 (4-vizinhos).
    - 'laplacian_8': Kernel Laplaciano 3x3 com 8-vizinhos (mais pronunciado).
    - 'high_boost': Aguçamento de alto ganho (High-Boost Filtering, A=1.5).
    - 'pil_sharpen': Filtro de aguçamento nativo do PIL (3x3 normalizado).
    """
    def __init__(
        self,
        method: str = "unsharp",
        radius: float = 1.5,
        percent: int = 150,
        threshold: int = 3
    ):
        self.method = method.lower()
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

        if self.method == "unsharp":
            self.filter_obj = ImageFilter.UnsharpMask(
                radius=self.radius,
                percent=self.percent,
                threshold=self.threshold
            )
        elif self.method == "laplacian":
            # Kernel Laplaciano 3x3 (centro=5, 4 vizinhos=-1)
            self.filter_obj = ImageFilter.Kernel(
                (3, 3),
                (0, -1, 0, -1, 5, -1, 0, -1, 0),
                scale=1,
                offset=0
            )
        elif self.method == "laplacian_8":
            # Kernel Laplaciano 3x3 (centro=9, 8 vizinhos=-1)
            self.filter_obj = ImageFilter.Kernel(
                (3, 3),
                (-1, -1, -1, -1, 9, -1, -1, -1, -1),
                scale=1,
                offset=0
            )
        elif self.method == "high_boost":
            # High-Boost com ganho no centro
            self.filter_obj = ImageFilter.Kernel(
                (3, 3),
                (-1, -1, -1, -1, 10, -1, -1, -1, -1),
                scale=2,
                offset=0
            )
        elif self.method == "pil_sharpen":
            self.filter_obj = ImageFilter.SHARPEN
        else:
            raise ValueError(
                f"Método de aguçamento '{method}' não suportado. "
                f"Opções: 'unsharp', 'laplacian', 'laplacian_8', 'high_boost', 'pil_sharpen'"
            )

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(self.filter_obj)

    def __repr__(self) -> str:
        if self.method == "unsharp":
            return (f"SharpenTransform(method='unsharp', radius={self.radius}, "
                    f"percent={self.percent}%, threshold={self.threshold})")
        return f"SharpenTransform(method='{self.method}')"


def get_sharpen_transform(
    method: str = "unsharp",
    radius: float = 1.5,
    percent: int = 150,
    threshold: int = 3,
    img_height: int = IMG_HEIGHT,
    img_width: int = IMG_WIDTH
) -> transforms.Compose:
    """
    Retorna o pipeline torchvision.transforms.Compose incluindo o filtro de aguçamento
    selecionado antes da conversão para Tensor e normalização ImageNet.
    """
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        SharpenTransform(method=method, radius=radius, percent=percent, threshold=threshold),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def apply_sharpening_to_pil(
    img: Image.Image,
    method: str = "unsharp",
    radius: float = 1.5,
    percent: int = 150,
    threshold: int = 3
) -> Tuple[Image.Image, np.ndarray]:
    """
    Aplica o filtro a uma imagem PIL e retorna (imagem_aguçada, mapa_diferença_absoluta).
    """
    transformer = SharpenTransform(method=method, radius=radius, percent=percent, threshold=threshold)
    img_sharp = transformer(img)
    arr_orig = np.array(img, dtype=np.float32)
    arr_sharp = np.array(img_sharp, dtype=np.float32)
    diff = np.abs(arr_sharp - arr_orig)
    return img_sharp, diff
