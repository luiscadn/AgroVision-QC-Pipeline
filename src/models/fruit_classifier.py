"""
fruit_classifier.py
Modelo MobileNetV2 con Transfer Learning para clasificar TIPO de fruta (6 clases).

Por qué MobileNetV2 en lugar de FruitQualityCNN:
- Ya fue entrenado en ImageNet (1.2M imágenes), conoce formas y texturas de frutas.
- Solo se ajusta la última capa para las 6 clases del proyecto.
- Corre eficientemente en CPU gracias a su arquitectura depthwise-separable.
- Logra 90-98% de accuracy en clasificación de frutas con pocas épocas de fine-tuning.

Orden de clases (ImageFolder, alfabético):
    0=banano | 1=granada | 2=guayaba | 3=limon | 4=manzana | 5=naranja
"""

import torch
import torch.nn as nn
from torchvision import models


def build_fruit_classifier(num_classes: int = 6, freeze_backbone: bool = True) -> nn.Module:
    """
    Construye un clasificador de tipo de fruta basado en MobileNetV2 preentrenado.

    Estrategia de Transfer Learning:
        - Etapa 1 (freeze_backbone=True):  Solo entrena la cabeza clasificadora (rápido, ~2-3 épocas).
        - Etapa 2 (freeze_backbone=False): Fine-tuning de toda la red (opcional, mayor precisión).

    Args:
        num_classes: Número de tipos de fruta (6 por defecto).
        freeze_backbone: Si True, congela los pesos del backbone y solo entrena la cabeza.

    Returns:
        nn.Module: Modelo listo para entrenar.
    """
    # Cargar MobileNetV2 con pesos preentrenados en ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congelar backbone si se indica (Transfer Learning clásico)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Reemplazar la cabeza clasificadora original (1000 clases ImageNet)
    # por una nueva adaptada a nuestras 6 frutas
    in_features = model.classifier[1].in_features   # 1280 en MobileNetV2
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    return model


def load_fruit_classifier(
    checkpoint_path: str,
    num_classes: int = 6,
    device: torch.device = None,
) -> nn.Module:
    """
    Carga un FruitClassifier (MobileNetV2) desde un checkpoint .pth.

    Args:
        checkpoint_path: Ruta al archivo .pth generado por train_fruit_cnn.py.
        num_classes: Número de clases (debe coincidir con el entrenamiento).
        device: Dispositivo destino. Si None, se usa CPU.

    Returns:
        nn.Module en modo eval, listo para inferencia.
    """
    if device is None:
        device = torch.device("cpu")

    model = build_fruit_classifier(num_classes=num_classes, freeze_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
