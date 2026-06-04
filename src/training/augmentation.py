from torchvision import transforms


# --- Transformaciones para el conjunto de ENTRENAMIENTO ---
# Incluyen técnicas de Data Augmentation para mejorar la generalización
# y reducir el sobreajuste (overfitting).
train_transforms = transforms.Compose([

    # Redimensionar imagen a 128x128
    transforms.Resize((128, 128)),

    # Rotar aleatoriamente hasta 20 grados
    transforms.RandomRotation(20),

    # Voltear horizontalmente con probabilidad 0.5
    transforms.RandomHorizontalFlip(),

    # Voltear verticalmente con probabilidad 0.2
    transforms.RandomVerticalFlip(p=0.2),

    # Cambiar brillo, contraste, saturación y matiz
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05
    ),

    # Convertir a tensor (escala [0, 1])
    transforms.ToTensor(),

    # Normalización estándar ImageNet para acelerar convergencia
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --- Transformaciones para VALIDACIÓN y TEST ---
# Sin aumentación aleatoria para medir el rendimiento real de forma limpia.
val_transforms = transforms.Compose([

    # Solo redimensionamiento a 128x128
    transforms.Resize((128, 128)),

    # Convertir a tensor
    transforms.ToTensor(),

    # Misma normalización que en train para coherencia
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])