import torch
from PIL import Image
from torchvision import transforms

from src.models.cnn_model import FruitQualityCNN


# Clases de calidad
classes = [
    "Mala",
    "Media",
    "Buena"
]


# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# Cargar modelo
model = FruitQualityCNN(num_classes=3)

model.load_state_dict(
    torch.load("experiments/checkpoints/best_model.pth")
)

model.eval()


# Cargar imagen
image = Image.open("test_image.jpg").convert("RGB")

# Aplicar transformaciones
image = transform(image)

# Agregar dimensión batch
image = image.unsqueeze(0)


# Desactivar gradientes
with torch.no_grad():

    outputs = model(image)

    predicted_class = torch.argmax(outputs, dim=1)

    prediction = classes[predicted_class.item()]


print(f"Predicción del modelo: {prediction}")