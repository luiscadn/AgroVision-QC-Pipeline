import torch

from cnn_model import FruitQualityCNN


# Crear el modelo
model = FruitQualityCNN(num_classes=3)

# Crear imagen falsa
fake_image = torch.randn(1, 3, 128, 128)

# Pasar imagen por la red
output = model(fake_image)

print("Salida del modelo:")
print(output)

print("\nForma de salida:")
print(output.shape)