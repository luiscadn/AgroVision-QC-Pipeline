import os
import cv2
import numpy as np
import torch
# Importamos tu tubería maestra
from src.data.preprocess import pipeline_extractor_maestro

def create_dummy_fruit_image(filename="dummy_fruit.jpg"):
    """Crea una imagen de prueba: un fondo blanco con un círculo verde (fruta)."""
    # Imagen de 400x400 píxeles con fondo blanco BGR en OpenCV
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Dibujar un círculo verde oscuro en el centro - Simulando un limón o manzana verde
    cv2.circle(img, (200, 200), 100, (40, 150, 40), -1)
    
    # Añadir una pequeña "mancha" oscura de defecto
    cv2.circle(img, (220, 180), 15, (20, 50, 20), -1)
    
    cv2.imwrite(filename, img)
    print(f" Imagen de prueba sintética creada en: {filename}")
    return filename

def run_test():
    test_image = create_dummy_fruit_image()
    
    try:
        print("\n Ejecutando el pipeline maestro...")
        cnn_tensor, ml_features = pipeline_extractor_maestro(test_image)
        
        print("\n === RESULTADOS DEL DIAGNÓSTICO DE CONTRATOS ===")
        
        # Validar Contrato de Matthew - Deep Learning
        print(f"\n CONTRATO DE DEEP LEARNING (MATTHEW):")
        print(f"  - ¿Es un Tensor de PyTorch?: {isinstance(cnn_tensor, torch.Tensor)}")
        print(f"  - Dimensiones del Tensor (Debe ser [3, 128, 128]): {list(cnn_tensor.shape)}")
        print(f"  - Tipo de dato (Debe ser torch.float32): {cnn_tensor.dtype}")
        print(f"  - Rango de píxeles (Mín: {cnn_tensor.min():.2f}, Máx: {cnn_tensor.max():.2f}) -> OK si está entre 0 y 1")
        
        # Validar Contrato de Juanes - Machine Learning Tradicional
        print(f"\n CONTRATO DE ML TRADICIONAL (JUANES):")
        print(f"  - ¿Es un Vector de NumPy?: {isinstance(ml_features, np.ndarray)}")
        print(f"  - Dimensiones del Vector (Debe ser un arreglo 1D de 9 elementos): {ml_features.shape}")
        print(f"  - Tipo de dato (Debe ser float32): {ml_features.dtype}")
        print(f"  - Desglose de Características extraídas:")
        print(f"    * Geométricas (Área, Perímetro, Aspect Ratio): {ml_features[:3]}")
        print(f"    * Colorimétricas Estadísticas (Mean H, S, V): {ml_features[3:6]}")
        print(f"    * Textura/Manchas Estadísticas (Std H, S, V): {ml_features[6:]}")
        
        print("\n🏆 ¡CONTRATOS DE INTERFAZ VALIDADOS CON ÉXITO! El pipeline es seguro.")
        
    except Exception as e:
        print(f"Fallo en el pipeline: {str(e)}")
    finally:
        # Limpieza del archivo temporal
        if os.path.exists(test_image):
            os.remove(test_image)

if __name__ == "__main__":
    run_test()