import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cnn_model import FruitQualityCNN


# Crear modelo
model = FruitQualityCNN(num_classes=3)

# Función de pérdida
criterion = nn.CrossEntropyLoss()

# Optimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simular imágenes falsas
fake_images = torch.randn(16, 3, 128, 128)

# Simular etiquetas falsas
fake_labels = torch.randint(0, 3, (16,))

# Número de epochs
epochs = 5

# Variable para guardar mejor loss
best_loss = float("inf")


for epoch in range(epochs):

    # Reiniciar gradientes
    optimizer.zero_grad()

    # Predicciones
    outputs = model(fake_images)

    # Calcular error
    loss = criterion(outputs, fake_labels)

    # Backpropagation
    loss.backward()

    # Actualizar pesos
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

    # Guardar mejor modelo
    if loss.item() < best_loss:

        best_loss = loss.item()

        torch.save(
            model.state_dict(),
            "experiments/checkpoints/best_model.pth"
        )

        print("Nuevo mejor modelo guardado")