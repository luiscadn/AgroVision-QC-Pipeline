import logging
import os

def setup_logger(name: str = "AgroVision") -> logging.Logger:
    """Configura el logger estándar para el proyecto."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Consola
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def get_project_root() -> str:
    """Devuelve la ruta absoluta del directorio raíz del proyecto."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
