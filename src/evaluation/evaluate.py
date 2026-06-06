import os
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.models.cnn_model import FruitQualityCNN
from src.training.augmentation import val_transforms
from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-Evaluation")


def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: str = "experiments/results/confusion_matrix.png"
):
    """
    Genera y guarda una imagen de la Matriz de Confusión con heatmap.

    Args:
        cm: Matriz de confusión como array de NumPy.
        class_names: Lista de nombres de las clases (ej: ['buena', 'mala', 'media']).
        output_path: Ruta donde se guardará la imagen PNG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Matriz de Confusión - Conjunto de Prueba",
        ylabel="Etiqueta Real",
        xlabel="Etiqueta Predicha"
    )

    # Anotar cada celda con el valor absoluto y el porcentaje
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Matriz de confusión guardada en: {output_path}")


def evaluate_model(
    processed_dir: str = "data/processed",
    checkpoints_dir: str = "experiments/checkpoints",
    results_dir: str = "experiments/results",
    batch_size: int = 32,
    num_classes: int = 3
):
    """
    Evalúa el rendimiento definitivo de la CNN entrenada exclusivamente
    sobre el conjunto de prueba (data/processed/test).

    Flujo:
        1. Carga el mejor modelo guardado (best_model.pth).
        2. Carga el conjunto de test con ImageFolder + DataLoader.
        3. Infiere las predicciones sobre todas las imágenes de prueba.
        4. Calcula Accuracy, Precision, Recall y F1-score (multiclase).
        5. Genera y guarda la Matriz de Confusión como imagen PNG.
        6. Guarda el reporte completo de métricas en un archivo JSON.

    Args:
        processed_dir: Directorio raíz de datos procesados (con split test/).
        checkpoints_dir: Directorio donde está guardado el checkpoint.
        results_dir: Directorio donde se guardan las métricas y gráficas.
        batch_size: Tamaño del lote para la inferencia.
        num_classes: Número de clases de calidad.

    Returns:
        dict: Diccionario con todas las métricas del conjunto de prueba.
    """
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, "best_model.pth")

    # ------------------------------------------------------------------
    # 1. Verificar que existe el checkpoint entrenado
    # ------------------------------------------------------------------
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint en '{checkpoint_path}'. "
            "Ejecuta primero el entrenamiento (train_cnn.py) para generar el modelo."
        )

    # ------------------------------------------------------------------
    # 2. Dispositivo de cómputo
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de evaluación: {device}")

    # ------------------------------------------------------------------
    # 3. Cargar el conjunto de prueba con ImageFolder
    # ------------------------------------------------------------------
    test_dir = os.path.join(processed_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"El directorio de prueba '{test_dir}' no existe. "
            "Ejecuta primero 'make_dataset.py'."
        )

    test_dataset = ImageFolder(root=test_dir, transform=val_transforms)
    if len(test_dataset) == 0:
        raise RuntimeError(
            f"No se encontraron imágenes en '{test_dir}'. "
            "Verifica que make_dataset.py haya procesado correctamente las imágenes."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    class_names = test_dataset.classes
    logger.info(f"Clases en test: {class_names}")
    logger.info(f"Total imágenes de prueba: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 4. Cargar el modelo entrenado
    # ------------------------------------------------------------------
    model = FruitQualityCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"Modelo cargado desde: {checkpoint_path}")

    # ------------------------------------------------------------------
    # 5. Inferencia sobre el conjunto de prueba
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ------------------------------------------------------------------
    # 6. Cálculo de métricas con scikit-learn
    # ------------------------------------------------------------------
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    )

    # ------------------------------------------------------------------
    # 7. Mostrar resultados en consola
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("MÉTRICAS FINALES EN EL CONJUNTO DE PRUEBA (TEST)")
    logger.info("=" * 60)
    logger.info(f"  Test Loss:        {test_loss:.4f}")
    logger.info(f"  Accuracy:         {accuracy:.2f}%")
    logger.info(f"  Precision (W):    {precision:.2f}%")
    logger.info(f"  Recall (W):       {recall:.2f}%")
    logger.info(f"  F1-Score (W):     {f1:.2f}%")
    logger.info("-" * 60)
    logger.info("Reporte por clase:\n" + report)

    # ------------------------------------------------------------------
    # 8. Guardar Matriz de Confusión como imagen
    # ------------------------------------------------------------------
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    _save_confusion_matrix(cm, class_names, output_path=cm_path)

    # ------------------------------------------------------------------
    # 9. Exportar métricas a JSON
    # ------------------------------------------------------------------
    metrics = {
        "test_loss": round(test_loss, 4),
        "accuracy_%": round(accuracy, 2),
        "precision_weighted_%": round(precision, 2),
        "recall_weighted_%": round(recall, 2),
        "f1_score_weighted_%": round(f1, 2),
        "num_test_images": len(test_dataset),
        "class_names": class_names,
        "confusion_matrix": cm.tolist()
    }

    metrics_path = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas del test guardadas en: {metrics_path}")

    return metrics


def evaluate_fruit_model(
    processed_dir: str = "data/processed_fruit",
    checkpoints_dir: str = "experiments/checkpoints",
    results_dir: str = "experiments/results",
    batch_size: int = 32,
    num_classes: int = 6
):
    """
    Evalúa el rendimiento definitivo de la CNN de frutas entrenada
    sobre el conjunto de prueba (data/processed_fruit/test).
    """
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, "best_fruit_model.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint de frutas en '{checkpoint_path}'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de evaluación de frutas: {device}")

    test_dir = os.path.join(processed_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"El directorio de prueba '{test_dir}' no existe."
        )

    test_dataset = ImageFolder(root=test_dir, transform=val_transforms)
    if len(test_dataset) == 0:
        raise RuntimeError(
            f"No se encontraron imágenes en '{test_dir}'."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    class_names = test_dataset.classes
    logger.info(f"Clases de fruta en test: {class_names}")
    logger.info(f"Total imágenes de prueba de fruta: {len(test_dataset)}")

    model = FruitQualityCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"Modelo de fruta cargado desde: {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    )

    logger.info("=" * 60)
    logger.info("MÉTRICAS FINALES DE LA CNN DE FRUTAS EN TEST")
    logger.info("=" * 60)
    logger.info(f"  Test Loss:        {test_loss:.4f}")
    logger.info(f"  Accuracy:         {accuracy:.2f}%")
    logger.info(f"  Precision (W):    {precision:.2f}%")
    logger.info(f"  Recall (W):       {recall:.2f}%")
    logger.info(f"  F1-Score (W):     {f1:.2f}%")
    logger.info("-" * 60)
    logger.info("Reporte por clase de fruta:\n" + report)

    cm_path = os.path.join(results_dir, "confusion_matrix_fruit_cnn.png")
    _save_confusion_matrix(cm, class_names, output_path=cm_path)

    metrics = {
        "test_loss": round(test_loss, 4),
        "accuracy_%": round(accuracy, 2),
        "precision_weighted_%": round(precision, 2),
        "recall_weighted_%": round(recall, 2),
        "f1_score_weighted_%": round(f1, 2),
        "num_test_images": len(test_dataset),
        "class_names": class_names,
        "confusion_matrix": cm.tolist()
    }

    metrics_path = os.path.join(results_dir, "fruit_test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas de frutas del test guardadas en: {metrics_path}")

    return metrics
