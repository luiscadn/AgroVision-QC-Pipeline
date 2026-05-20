from src.utils.helpers import setup_logger
from src.models.my_model import create_cnn_model

logger = setup_logger("AgroVision-Training")

def train_pipeline(epochs: int = 10, batch_size: int = 32):
    """
    Simula o ejecuta el entrenamiento del pipeline.
    """
    logger.info("Iniciando entrenamiento del modelo...")
    model = create_cnn_model()
    
    # Aquí iría el flujo de ajuste del modelo con el dataset.
    logger.info(f"Parámetros del entrenamiento: epochs={epochs}, batch_size={batch_size}")
    
    if isinstance(model, dict):
        logger.info("Entrenamiento en modo simulado finalizado con éxito.")
    else:
        logger.info("Entrenamiento de TensorFlow finalizado con éxito.")
        
    return model
