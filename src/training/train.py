from src.utils.helpers import setup_logger
from src.training.train_cnn import train_cnn

logger = setup_logger("AgroVision-Training")


def train_pipeline(
    processed_dir: str = "data/processed",
    checkpoints_dir: str = "experiments/checkpoints",
    results_dir: str = "experiments/results",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 3
):
    """
    Punto de entrada unificado para el entrenamiento de la CNN.
    Delega la ejecución real a train_cnn.py (PyTorch + datos reales).

    Args:
        processed_dir: Directorio raíz con los splits train/val/test procesados.
        checkpoints_dir: Directorio donde se guarda el mejor checkpoint.
        results_dir: Directorio donde se guardan las curvas y el historial.
        epochs: Número de épocas de entrenamiento.
        batch_size: Tamaño del lote.
        learning_rate: Tasa de aprendizaje del optimizador Adam.
        num_classes: Número de clases de calidad (buena, media, mala = 3).

    Returns:
        dict: Historial de métricas de entrenamiento y validación por época.
    """
    logger.info("Iniciando entrenamiento del modelo CNN con datos reales...")

    history = train_cnn(
        processed_dir=processed_dir,
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_classes=num_classes
    )

    logger.info("Entrenamiento finalizado exitosamente.")
    return history
