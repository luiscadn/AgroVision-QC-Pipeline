import torch

from src.models.cnn_model import FruitQualityCNN


# Crear modelo
model = FruitQualityCNN(num_classes=3)

# Cargar pesos entrenados
model.load_state_dict(
    torch.load("experiments/checkpoints/best_model.pth")
)

# Modo evaluación
model.eval()

print("Modelo cargado correctamente")