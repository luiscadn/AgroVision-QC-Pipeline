import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.fruit_classifier import build_fruit_classifier
from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-TrainFruitCNN")

# ── Transformaciones 224x224 requeridas por MobileNetV2 ──────────────────────
FRUIT_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

FRUIT_VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_dataloaders(
    processed_dir: str = "data/processed_fruit",
    batch_size: int = 32,
    num_workers: int = 0
):
    """
    Construye los DataLoaders con imágenes 224x224 para MobileNetV2.
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

    train_dataset = ImageFolder(root=train_dir, transform=FRUIT_TRAIN_TRANSFORMS)
    val_dataset   = ImageFolder(root=val_dir,   transform=FRUIT_VAL_TRANSFORMS)

    if len(train_dataset) == 0:
        raise RuntimeError(f"No se encontraron imágenes en '{train_dir}'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    logger.info(f"Clases detectadas (orden ImageFolder): {class_names}")
    logger.info(f"Imágenes de entrenamiento: {len(train_dataset)}")
    logger.info(f"Imágenes de validación:   {len(val_dataset)}")

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
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 6
):
    """
    Entrena MobileNetV2 con Transfer Learning para clasificar tipo de fruta.

    Estrategia en 2 fases:
      Fase 1 (primeras epochs//2):  backbone congelado, solo entrena la cabeza.
      Fase 2 (segundas epochs//2):  fine-tuning de toda la red con lr reducido.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    logger.info("Modelo: MobileNetV2 (Transfer Learning desde ImageNet)")

    train_loader, val_loader, class_names = build_dataloaders(
        processed_dir=processed_dir,
        batch_size=batch_size
    )

    # ── Fase 1: backbone congelado ────────────────────────────────────────────
    phase1_epochs = max(1, epochs // 2)
    phase2_epochs = epochs - phase1_epochs

    model = build_fruit_classifier(num_classes=num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Solo optimizar la cabeza en Fase 1
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    final_checkpoint_path = os.path.join(checkpoints_dir, "best_fruit_model.pth")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "class_names": class_names
    }

    logger.info(f"=== FASE 1: Cabeza clasificadora ({phase1_epochs} épocas, backbone congelado) ===")
    logger.info("-" * 60)

    best_val_loss = _run_epochs(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        phase1_epochs, epochs, 0, device, history, final_checkpoint_path, best_val_loss
    )

    # ── Fase 2: fine-tuning de toda la red ───────────────────────────────────
    if phase2_epochs > 0:
        logger.info(f"=== FASE 2: Fine-tuning completo ({phase2_epochs} épocas, lr reducido) ===")
        logger.info("-" * 60)

        # Descongelar backbone
        for param in model.parameters():
            param.requires_grad = True

        # lr más bajo para no destruir los pesos preentrenados
        optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        best_val_loss = _run_epochs(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            phase2_epochs, epochs, phase1_epochs, device, history, final_checkpoint_path, best_val_loss
        )

    # ── Guardar curvas ────────────────────────────────────────────────────────
    curves_path = os.path.join(results_dir, "learning_curves_fruit.png")
    _save_learning_curves(history, output_path=curves_path)

    history_path = os.path.join(results_dir, "training_history_fruit.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Entrenamiento tipo-fruta finalizado. Mejor val_loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint guardado en: {final_checkpoint_path}")
    logger.info(f"Orden de clases: {class_names}")
    return history


def _run_epochs(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs, total_epochs, epoch_offset, device, history, checkpoint_path, best_val_loss
):
    """Bucle de entrenamiento/validación reutilizable."""
    for i in range(num_epochs):
        epoch = epoch_offset + i

        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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
        train_acc  = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc,  2))
        history["val_loss"].append(round(val_loss,   4))
        history["val_acc"].append(round(val_acc,    2))

        is_best = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            is_best = " ✓ Mejor modelo guardado"

        logger.info(
            f"Época [{epoch+1:02d}/{total_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            f"{is_best}"
        )

    return best_val_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenar CNN clasificador de tipo de fruta")
    parser.add_argument("--epochs",               type=int,   default=10,                    help="Épocas (default: 10)")
    parser.add_argument("--batch_size",           type=int,   default=32,                    help="Batch size (default: 32)")
    parser.add_argument("--lr",                   type=float, default=0.001,                 help="Learning rate (default: 0.001)")
    parser.add_argument("--processed_fruit_dir",  type=str,   default="data/processed_fruit", help="Directorio con splits de tipo de fruta")
    args = parser.parse_args()

    train_fruit_cnn(
        processed_dir=args.processed_fruit_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
