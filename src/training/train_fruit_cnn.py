import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.cnn_model import FruitQualityCNN
from src.training.augmentation import train_transforms, val_transforms
from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-TrainFruitCNN")

def build_dataloaders(
    processed_dir: str = "data/processed_fruit",
    batch_size: int = 32,
    num_workers: int = 0
):
    """
    Construye los DataLoaders de PyTorch para los splits de train y val de frutas.
    """
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"El directorio de entrenamiento '{train_dir}' no existe. "
            "Ejecuta primero 'make_dataset.py' para generar los datos procesados."
        )
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"El directorio de validación '{val_dir}' no existe. "
            "Ejecuta primero 'make_dataset.py' para generar los datos procesados."
        )

    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=val_transforms)

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No se encontraron imágenes en '{train_dir}'."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    class_names = train_dataset.classes
    logger.info(f"Clases detectadas: {class_names}")
    logger.info(f"Imágenes de entrenamiento: {len(train_dataset)}")
    logger.info(f"Imágenes de validación: {len(val_dataset)}")

    return train_loader, val_loader, class_names

def _save_learning_curves(history: dict, output_path: str = "experiments/results/learning_curves_fruit.png"):
    """
    Genera y guarda un gráfico de las curvas de pérdida y precisión para frutas.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Aprendizaje - CNN Frutas AgroVision", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Pérdida por Época (Loss)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=4)
    axes[1].set_title("Precisión por Época (Accuracy)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Curvas de aprendizaje de fruta guardadas en: {output_path}")

def train_fruit_cnn(
    processed_dir: str = "data/processed_fruit",
    checkpoints_dir: str = "experiments/checkpoints",
    results_dir: str = "experiments/results",
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 6
):
    """
    Entrena la CNN de tipo de fruta sobre el dataset estructurado en data/processed_fruit.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de entrenamiento de frutas: {device}")

    train_loader, val_loader, class_names = build_dataloaders(
        processed_dir=processed_dir,
        batch_size=batch_size
    )

    model = FruitQualityCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    # Use epoch-specific checkpoint filenames to avoid file lock conflicts
    checkpoint_path_template = os.path.join(checkpoints_dir, "best_fruit_model_epoch_{epoch}.pth")
    # Final best model will be saved as 'best_fruit_model.pth' after training completes
    final_checkpoint_path = os.path.join(checkpoints_dir, "best_fruit_model.pth")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "class_names": class_names
    }

    logger.info(f"Iniciando entrenamiento de frutas: {epochs} épocas, batch={batch_size}, lr={learning_rate}")
    logger.info("-" * 60)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        is_best = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoint for this epoch
            epoch_checkpoint_path = checkpoint_path_template.format(epoch=epoch + 1)
            torch.save(model.state_dict(), epoch_checkpoint_path)
            # Also update the final checkpoint path to the latest best model
            torch.save(model.state_dict(), final_checkpoint_path)
            is_best = " ✓ Mejor modelo de fruta guardado"

        logger.info(
            f"Época [{epoch + 1:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            f"{is_best}"
        )

    curves_path = os.path.join(results_dir, "learning_curves_fruit.png")
    _save_learning_curves(history, output_path=curves_path)

    history_path = os.path.join(results_dir, "training_history_fruit.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Historial de entrenamiento de frutas guardado en: {history_path}")

    logger.info("=" * 60)
    logger.info(f"Entrenamiento de frutas finalizado. Mejor val_loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint guardado en: {final_checkpoint_path}")

    return history

if __name__ == "__main__":
    train_fruit_cnn()
