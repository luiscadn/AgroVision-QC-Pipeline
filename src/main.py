import os
import sys

# Asegurar que el directorio src está en el PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import setup_logger
from src.training.train import train_pipeline
from src.evaluation.evaluate import evaluate_model

logger = setup_logger("AgroVision-Main")

def main():
    logger.info("=== AgroVision QC Pipeline ===")
    logger.info("Fase 1: Preparación y carga de datos")
    
    logger.info("Fase 2: Entrenamiento del modelo")
    model = train_pipeline(epochs=5, batch_size=16)
    
    logger.info("Fase 3: Evaluación")
    metrics = evaluate_model(model)
    
    logger.info("Fase 4: Guardado de resultados")
    # Los checkpoints se guardarían en experiments/checkpoints/
    logger.info("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()
