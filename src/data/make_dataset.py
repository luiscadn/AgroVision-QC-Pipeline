import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path

from src.utils.helpers import setup_logger
from src.data.preprocess import load_and_segment_fruit, get_traditional_features

logger = setup_logger("AgroVision-MakeDataset")

def ensure_directories_exist(base_dir: str, classes: list, splits: list = ['train', 'val', 'test']):
    """
    Construye la estructura de carpetas requerida en el directorio de destino.
    """
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
            
    # Garantizar que el directorio de resultados de experimentos exista
    os.makedirs("experiments/results", exist_ok=True)

def get_stratified_split(class_images: list, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    """
    Realiza una partición aleatoria estratificada a nivel de clase.
    """
    # Semilla fija para reproducibilidad
    random.seed(42)
    random.shuffle(class_images)
    
    total = len(class_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    return {
        'train': class_images[:train_end],
        'val': class_images[train_end:val_end],
        'test': class_images[val_end:]
    }

def process_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """
    Orquestador para el procesamiento masivo de imágenes (Fase 3: CRISP-DM).
    Itera sobre los datos crudos, aplica la lógica de preprocesamiento,
    exporta las imágenes para la CNN y genera el CSV para ML tradicional.
    """
    logger.info("Iniciando procesamiento masivo de datos (Make Dataset)...")
    
    if not os.path.exists(raw_dir):
        logger.error(f"El directorio de origen {raw_dir} no existe. Por favor, asegúrate de colocar las imágenes crudas allí.")
        return

    # Escanear clases disponibles (carpetas dentro de data/raw)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    if not classes:
        logger.warning(f"No se detectaron carpetas de clases en {raw_dir}.")
        return
        
    logger.info(f"Clases identificadas para procesar: {classes}")
    
    # Asegurarnos de tener las carpetas necesarias en data/processed
    ensure_directories_exist(processed_dir, classes)
    
    # Matriz para ir acumulando todas las características para Scikit-Learn
    all_tabular_features = []
    
    # Extensiones de imagen soportadas
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    # Bucle por clase
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        
        images = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not images:
            logger.warning(f"No se encontraron imágenes válidas en la clase '{cls}'")
            continue
            
        logger.info(f"➤ Procesando clase '{cls}' ({len(images)} imágenes registradas)...")
        
        # Realizar el split (70/15/15)
        splits = get_stratified_split(images)
        
        processed_count = {'train': 0, 'val': 0, 'test': 0}
        error_count = 0
        
        # Iterar sobre las particiones
        for split_name, split_imgs in splits.items():
            for img_name in split_imgs:
                img_path = os.path.join(cls_dir, img_name)
                dest_path = os.path.join(processed_dir, split_name, cls, img_name)
                
                try:
                    # 1. Ejecutar Fase A: Carga y Segmentación
                    # Devuelve la imagen recortada en RGB
                    img_cropped, contour = load_and_segment_fruit(img_path)
                    
                    # 2. Contrato Deep Learning (Matthew): Guardar en disco la imagen procesada
                    # Redimensionamos estáticamente a 128x128 tal como espera la CNN
                    img_resized = cv2.resize(img_cropped, (128, 128), interpolation=cv2.INTER_LINEAR)
                    
                    # IMPORTANTE: OpenCV trabaja con BGR por defecto para guardar y mostrar imágenes.
                    # img_cropped está en RGB, por lo que revertimos a BGR para guardar correctamente.
                    img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dest_path, img_bgr_to_save)
                    
                    # 3. Contrato ML Tradicional (Juanes): Extracción de Características
                    # Obtenemos el vector de 9 elementos - 1D numpy array
                    ml_features = get_traditional_features(img_cropped, contour)
                    
                    # Transformar a lista de Python estándar y concatenar la etiqueta - label
                    feature_row = ml_features.tolist() + [cls]
                    all_tabular_features.append(feature_row)
                    
                    processed_count[split_name] += 1
                    
                except Exception as e:
                    # Robustez: Manejar archivos corruptos o fallos de segmentación para no tumbar todo el bucle
                    logger.error(f"Fallo aislado procesando imagen {img_path}: {str(e)}")
                    error_count += 1
                    
        total_processed = sum(processed_count.values())
        logger.info(f" Clase '{cls}' completada. Procesadas: {total_processed}/{len(images)} (Train:{processed_count['train']} | Val:{processed_count['val']} | Test:{processed_count['test']} | Errores:{error_count})")
        
    # --- EXPORTAR MATRIZ DE DISEÑO PARA ML ---
    csv_path = "experiments/results/features_traditional_ml.csv"
    logger.info(f"Exportando matriz tabular de características a: {csv_path}")
    
    headers = [
        'area', 'perimeter', 'aspect_ratio', 
        'h_mean', 's_mean', 'v_mean', 
        'h_std', 's_std', 'v_std', 'label'
    ]
    
    try:
        # Usamos pandas para una escritura y estructura de datos óptima
        df_features = pd.DataFrame(all_tabular_features, columns=headers)
        df_features.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Exportación de CSV completada. Registros exportados: {len(df_features)}")
    except Exception as e:
        logger.error(f"Ocurrió un error crítico exportando el archivo CSV: {str(e)}")
        
    logger.info("== PROCESAMIENTO MASIVO FINALIZADO EXITOSAMENTE ==")

if __name__ == "__main__":
    process_dataset()
