import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import setup_logger
from src.data.make_dataset import process_dataset
from src.training.train_cnn import train_cnn
from src.evaluation.evaluate import evaluate_model
from src.data.extract_dataset import extract_and_structure_zip

logger = setup_logger("AgroVision-Main")

# ---- Configuración del pipeline ----
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CHECKPOINTS_DIR = "experiments/checkpoints"
RESULTS_DIR = "experiments/results"

# Hiperparámetros de entrenamiento
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # buena, media, mala


def _processed_data_exists(processed_dir: str) -> bool:
    """
    Verifica si ya existen imágenes procesadas en los tres splits (train/val/test).
    Retorna True si los tres directorios existen y contienen al menos una imagen.
    """
    for split in ["train", "val", "test"]:
        split_path = os.path.join(processed_dir, split)
        if not os.path.isdir(split_path):
            return False
        # Verificar que tenga al menos una imagen (en cualquier subcarpeta)
        has_images = any(
            fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            for _, _, files in os.walk(split_path)
            for fname in files
        )
        if not has_images:
            return False
    return True


def main():
    logger.info("=" * 60)
    logger.info("      AgroVision QC Pipeline - Inicio del Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # FASE 1: Preparación de Datos
    # Sólo ejecuta make_dataset si los datos procesados aún no existen.
    # Asume que data/raw/ ya contiene las imágenes originales o busca un ZIP.
    # ------------------------------------------------------------------
    logger.info("\n[Fase 1] Verificando datos procesados...")

    if _processed_data_exists(PROCESSED_DIR):
        logger.info(
            f"Los datos procesados ya existen en '{PROCESSED_DIR}'. "
            "Se omite el preprocesamiento."
        )
    else:
        logger.info(
            f"No se encontraron datos procesados en '{PROCESSED_DIR}'."
        )
        
        # Intentar extraer desde un ZIP en la raíz si existe
        extract_and_structure_zip()
        
        if not os.path.isdir(RAW_DIR):
            logger.error(
                f"El directorio de datos crudos '{RAW_DIR}' no existe. "
                "Por favor, coloca las imágenes originales allí con la estructura:\n"
                "  data/raw/<calidad>/<nombre_imagen>.jpg\n"
                "  (calidad puede ser: buena, media, mala)\n"
                "Alternativamente, coloca el archivo ZIP del dataset (ej: Fruits.zip) en la raíz."
            )
            sys.exit(1)
            
        logger.info(f"Iniciando preprocesamiento desde '{RAW_DIR}'...")
        process_dataset(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

    # ------------------------------------------------------------------
    # FASE 2: Entrenamiento de la CNN con datos reales
    # ------------------------------------------------------------------
    logger.info("\n[Fase 2] Entrenando la CNN con datos reales...")
    history = train_cnn(
        processed_dir=PROCESSED_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_classes=NUM_CLASSES
    )

    # Diagnóstico rápido de overfitting en consola
    if len(history["train_loss"]) > 1:
        last_train_loss = history["train_loss"][-1]
        last_val_loss = history["val_loss"][-1]
        gap = last_val_loss - last_train_loss
        if gap > 0.3:
            logger.warning(
                f"Posible SOBREAJUSTE detectado: Val Loss ({last_val_loss:.4f}) supera "
                f"Train Loss ({last_train_loss:.4f}) por {gap:.4f}. "
                "Considera aumentar la intensidad de Data Augmentation o añadir más Dropout."
            )
        else:
            logger.info(
                f"Sin señales claras de sobreajuste "
                f"(Train Loss: {last_train_loss:.4f} | Val Loss: {last_val_loss:.4f})."
            )

    # ------------------------------------------------------------------
    # FASE 2b: Entrenamiento de Modelos de ML Tradicionales
    # ------------------------------------------------------------------
    logger.info("\n[Fase 2b] Entrenando modelos de Machine Learning tradicionales (SVM y Random Forest)...")
    from src.training.train_traditional_ml import train_traditional_ml
    train_traditional_ml(
        features_csv=os.path.join(RESULTS_DIR, "features_traditional_ml.csv"),
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR
    )

    # ------------------------------------------------------------------
    # FASE 3: Evaluación en el Conjunto de Prueba
    # ------------------------------------------------------------------
    logger.info("\n[Fase 3] Evaluando el modelo en el conjunto de prueba (test)...")
    metrics = evaluate_model(
        processed_dir=PROCESSED_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES
    )

    # ------------------------------------------------------------------
    # RESUMEN FINAL
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("         RESUMEN FINAL DEL PIPELINE")
    logger.info("=" * 60)
    logger.info("  [DEEP LEARNING - CNN]")
    logger.info(f"    Accuracy en Test:        {metrics['accuracy_%']:.2f}%")
    logger.info(f"    Checkpoint:              {CHECKPOINTS_DIR}/best_model.pth")
    logger.info(f"    Curvas de aprendizaje:   {RESULTS_DIR}/learning_curves.png")
    logger.info(f"    Matriz de confusión CNN: {RESULTS_DIR}/confusion_matrix.png")
    logger.info(f"    Métricas JSON:           {RESULTS_DIR}/test_metrics.json")
    logger.info("  [MACHINE LEARNING TRADICIONAL]")
    logger.info(f"    Modelos (SVM, RF):       {CHECKPOINTS_DIR}/svm_model.pkl, {CHECKPOINTS_DIR}/random_forest_model.pkl")
    logger.info(f"    Matrices Confusión ML:   {RESULTS_DIR}/confusion_matrix_svm.png, {RESULTS_DIR}/confusion_matrix_rf.png")
    logger.info(f"    Métricas JSON ML:        {RESULTS_DIR}/traditional_ml_metrics.json")
    logger.info("=" * 60)
    logger.info("Pipeline completado exitosamente.")


if __name__ == "__main__":
    main()
