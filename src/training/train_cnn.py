import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI (compatible con servidores/consola)
import matplotlib.pyplot as plt

from src.models.cnn_model import FruitQualityCNN
from src.training.augmentation import train_transforms, val_transforms
from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-TrainCNN")


def build_dataloaders(
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    num_workers: int = 0
):
    """
    Construye los DataLoaders de PyTorch para los splits de train y val
    usando ImageFolder, que asume la estructura:
        data/processed/train/buena/
        data/processed/train/media/
        data/processed/train/mala/
        data/processed/val/buena/
        ...

    Args:
        processed_dir: Ruta al directorio de imágenes procesadas.
        batch_size: Tamaño del lote.
        num_workers: Hilos paralelos para la carga de datos.

    Returns:
        tuple: (train_loader, val_loader, class_names)
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
            f"No se encontraron imágenes en '{train_dir}'. "
            "Verifica que make_dataset.py haya procesado correctamente las imágenes."
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


def _save_learning_curves(history: dict, output_path: str = "experiments/results/learning_curves.png"):
    """
    Genera y guarda un gráfico de las curvas de pérdida y precisión
    por época para los conjuntos de entrenamiento y validación.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Aprendizaje - CNN AgroVision", fontsize=14, fontweight="bold")

    # --- Curva de Pérdida (Loss) ---
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Pérdida por Época (Loss)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Curva de Precisión (Accuracy) ---
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
    logger.info(f"Curvas de aprendizaje guardadas en: {output_path}")


def train_cnn(
    processed_dir: str = "data/processed",
    checkpoints_dir: str = "experiments/checkpoints",
    results_dir: str = "experiments/results",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 3
):
    """
    Entrena la FruitQualityCNN usando datos reales de PyTorch ImageFolder.
    Al finalizar cada época, evalúa el modelo sobre el conjunto de validación.
    Guarda el mejor modelo (menor val_loss) y las curvas de aprendizaje.

    Args:
        processed_dir: Directorio raíz de datos procesados (con splits train/val/test).
        checkpoints_dir: Directorio donde se guarda el mejor checkpoint.
        results_dir: Directorio donde se guardan las gráficas y métricas.
        epochs: Número de épocas de entrenamiento.
        batch_size: Tamaño del lote.
        learning_rate: Tasa de aprendizaje inicial del optimizador Adam.
        num_classes: Número de clases de calidad (buena, media, mala = 3).

    Returns:
        dict: Historial de métricas por época.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Dispositivo de cómputo (GPU/MPS si está disponible, si no CPU)
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info(f"Dispositivo de entrenamiento: {device}")

    # ------------------------------------------------------------------
    # 2. Cargar los DataLoaders con datos REALES
    # ------------------------------------------------------------------
    train_loader, val_loader, class_names = build_dataloaders(
        processed_dir=processed_dir,
        batch_size=batch_size
    )

    # ------------------------------------------------------------------
    # 3. Modelo, función de pérdida balanceada y optimizador
    # ------------------------------------------------------------------
    # Obtener el dataset del DataLoader para calcular el desbalance de clases
    train_dataset = train_loader.dataset
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    
    # Calcular pesos con frecuencia inversa: weight_c = Total_Samples / (Num_Classes * Samples_c)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    logger.info(f"Frecuencias de clase en entrenamiento: {dict(zip(class_names, class_counts))}")
    logger.info(f"Pesos de balanceo calculados (Frecuencia Inversa): {dict(zip(class_names, class_weights.tolist()))}")

    model = FruitQualityCNN(num_classes=num_classes).to(device)
    # Aplicar la penalización de gradientes de forma matemática mediante el tensor de pesos
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler: reduce el lr si el val_loss no mejora en 5 épocas
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ------------------------------------------------------------------
    # 4. Ciclo de entrenamiento y validación
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(checkpoints_dir, "best_model.pth")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "class_names": class_names
    }

    logger.info(f"Iniciando entrenamiento: {epochs} épocas, batch={batch_size}, lr={learning_rate}")
    logger.info("-" * 60)

    for epoch in range(epochs):

        # ---- FASE DE ENTRENAMIENTO ----
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

        # ---- FASE DE VALIDACIÓN ----
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

        # Actualizar scheduler según el val_loss
        scheduler.step(val_loss)

        # Guardar historial
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        # Guardar mejor modelo
        is_best = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            is_best = " ✓ Mejor modelo guardado"

        logger.info(
            f"Época [{epoch + 1:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            f"{is_best}"
        )

    # ------------------------------------------------------------------
    # 5. Guardar curvas de aprendizaje
    # ------------------------------------------------------------------
    curves_path = os.path.join(results_dir, "learning_curves.png")
    _save_learning_curves(history, output_path=curves_path)

    # Guardar historial en JSON para consulta posterior
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Historial de entrenamiento guardado en: {history_path}")

    logger.info("=" * 60)
    logger.info(f"Entrenamiento finalizado. Mejor val_loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint guardado en: {checkpoint_path}")

    return history