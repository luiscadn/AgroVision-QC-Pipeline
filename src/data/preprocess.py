import cv2
import numpy as np
from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-Data")

def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Carga una imagen y la preprocesa (redimensionado y normalización).
    
    Args:
        image_path (str): Ruta de la imagen.
        target_size (tuple): Tamaño destino para la imagen.
        
    Returns:
        np.ndarray: Imagen preprocesada lista para el modelo.
    """
    logger.info(f"Cargando imagen desde: {image_path}")
    # Cargar imagen en BGR
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en {image_path}")
        
    # Convertir a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar
    img_resized = cv2.resize(img, target_size)
    
    # Normalizar valores de píxeles a [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized
