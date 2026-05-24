from torchvision import transforms


train_transforms = transforms.Compose([

    # Redimensionar imagen
    transforms.Resize((128, 128)),

    # Rotar aleatoriamente
    transforms.RandomRotation(20),

    # Voltear horizontalmente
    transforms.RandomHorizontalFlip(),

    # Cambiar brillo y contraste
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    # Convertir a tensor
    transforms.ToTensor()
])