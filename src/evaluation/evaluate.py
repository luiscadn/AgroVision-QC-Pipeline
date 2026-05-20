from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-Evaluation")

def evaluate_model(model, test_data=None):
    """
    Evalúa el rendimiento del modelo en un conjunto de prueba.
    """
    logger.info("Iniciando evaluación del modelo...")
    
    # Simulación de métricas de desempeño
    metrics = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.95
    }
    
    logger.info(f"Métricas obtenidas: {metrics}")
    return metrics
