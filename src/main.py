import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import setup_logger
from src.data.make_dataset import process_dataset
from src.training.train_cnn import train_cnn
from src.training.train_fruit_cnn import train_fruit_cnn
from src.evaluation.evaluate import evaluate_model, evaluate_fruit_model
from src.data.extract_dataset import extract_and_structure_zip

logger = setup_logger("AgroVision-Main")

# ---- Configuración del pipeline ----
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CHECKPOINTS_DIR = "experiments/checkpoints"
RESULTS_DIR = "experiments/results"

# Hiperparámetros de entrenamiento
EPOCHS = 2  # Limitado a 2 épocas para entrenamiento rápido
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # buena, media, mala
FRUIT_PROCESSED_DIR = "data/processed_fruit"
FRUIT_NUM_CLASSES = 6


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

    if _processed_data_exists(PROCESSED_DIR) and _processed_data_exists(FRUIT_PROCESSED_DIR):
        logger.info(
            f"Los datos procesados ya existen en '{PROCESSED_DIR}' y '{FRUIT_PROCESSED_DIR}'. "
            "Se omite el preprocesamiento."
        )
    else:
        logger.info(
            f"No se encontraron datos procesados completos."
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
        process_dataset(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, processed_fruit_dir=FRUIT_PROCESSED_DIR)

    # ------------------------------------------------------------------
    # FASE 2: Entrenamiento de la CNN con datos reales
    # ------------------------------------------------------------------
    logger.info("\n[Fase 2] Entrenando la CNN de Calidad con datos reales...")
    history = train_cnn(
        processed_dir=PROCESSED_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_classes=NUM_CLASSES
    )

    logger.info("\n[Fase 2a] Entrenando la CNN de Tipo de Fruta...")
    history_fruit = train_fruit_cnn(
        processed_dir=FRUIT_PROCESSED_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_classes=FRUIT_NUM_CLASSES
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
    logger.info("\n[Fase 3] Evaluando el modelo de calidad CNN en el conjunto de prueba...")
    metrics = evaluate_model(
        processed_dir=PROCESSED_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES
    )

    logger.info("\n[Fase 3a] Evaluando el modelo de fruta CNN en el conjunto de prueba...")
    metrics_fruit = evaluate_fruit_model(
        processed_dir=FRUIT_PROCESSED_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        batch_size=BATCH_SIZE,
        num_classes=FRUIT_NUM_CLASSES
    )

    # ------------------------------------------------------------------
    # RESUMEN FINAL
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("         RESUMEN FINAL DEL PIPELINE")
    logger.info("=" * 60)
    logger.info("  [DEEP LEARNING - CNN]")
    logger.info(f"    Quality Accuracy en Test:  {metrics['accuracy_%']:.2f}%")
    logger.info(f"    Fruit Type Accuracy:       {metrics_fruit['accuracy_%']:.2f}%")
    logger.info(f"    Checkpoint Calidad:        {CHECKPOINTS_DIR}/best_model.pth")
    logger.info(f"    Checkpoint Fruta:          {CHECKPOINTS_DIR}/best_fruit_model.pth")
    logger.info(f"    Curvas de aprendizaje:     {RESULTS_DIR}/learning_curves.png, {RESULTS_DIR}/learning_curves_fruit.png")
    logger.info("  [MACHINE LEARNING TRADICIONAL]")
    logger.info(f"    Modelos Calidad (SVM, RF): {CHECKPOINTS_DIR}/svm_model.pkl, {CHECKPOINTS_DIR}/random_forest_model.pkl")
    logger.info(f"    Modelo Fruta (ML):         {CHECKPOINTS_DIR}/fruit_type_model.pkl")
    logger.info("=" * 60)
    logger.info("Pipeline completado exitosamente.")


if __name__ == "__main__":
    main()
