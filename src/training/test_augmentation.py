from PIL import Image

from src.training.augmentation import train_transforms


# Abrir imagen de prueba
image = Image.open("test_image.jpg").convert("RGB")
# Aplicar augmentation
transformed_image = train_transforms(image)

print("Transformación aplicada correctamente")

print("Shape del tensor:")
print(transformed_image.shape)