import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path

from src.utils.helpers import setup_logger
from src.data.preprocess import load_and_segment_fruit, get_traditional_features

logger = setup_logger("AgroVision-MakeDataset")

# Mapeo oficial de calidad requerido por el modelo CNN y ML
QUALITY_MAP = {
    'buena': 0,
    'media': 1,
    'mala': 2
}

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
    Itera sobre los datos crudos, detecta las clases de calidad, aplica la lógica 
    de preprocesamiento, exporta las imágenes organizadas por calidad para la CNN 
    y genera el CSV con IDs numéricos para ML tradicional.
    """
    logger.info("Iniciando procesamiento masivo de datos (Make Dataset)...")
    
    if not os.path.exists(raw_dir):
        logger.error(f"El directorio de origen {raw_dir} no existe. Por favor, asegúrate de colocar las imágenes crudas allí.")
        return

    # Definir las clases objetivo de calidad (buena, media, mala)
    target_classes = list(QUALITY_MAP.keys())
    
    # Asegurarnos de tener las carpetas necesarias en data/processed organizadas por calidad
    ensure_directories_exist(processed_dir, target_classes)
    
    # Matriz para ir acumulando todas las características para Scikit-Learn
    all_tabular_features = []
    
    # Extensiones de imagen soportadas
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    # Recolectar y agrupar todas las imágenes por su calidad (buena, media, mala)
    # Soporta estructuras mixtas: data/raw/buena/manzana/img.jpg o data/raw/Manzana/buena_01.jpg
    images_by_quality = {cls: [] for cls in target_classes}
    
    logger.info("Escaneando recursivamente el directorio de origen en búsqueda de clases de calidad...")
    for path in Path(raw_dir).rglob('*'):
        if path.suffix.lower() in valid_exts:
            path_str = str(path).lower()
            
            # Inferir la clase de calidad analizando la ruta completa o el nombre del archivo
            if 'buena' in path_str:
                images_by_quality['buena'].append(path)
            elif 'media' in path_str:
                images_by_quality['media'].append(path)
            elif 'mala' in path_str:
                images_by_quality['mala'].append(path)
            else:
                # Omitir imágenes cuya calidad no se puede determinar
                pass
    
    # Bucle por clase de calidad
    for quality_cls, images in images_by_quality.items():
        if not images:
            logger.warning(f"No se encontraron imágenes detectadas para la clase de calidad '{quality_cls}'")
            continue
            
        logger.info(f"➤ Procesando clase de calidad '{quality_cls}' ({len(images)} imágenes registradas)...")
        
        # Realizar el split estratificado (70/15/15)
        splits = get_stratified_split(images)
        
        processed_count = {'train': 0, 'val': 0, 'test': 0}
        error_count = 0
        
        # Iterar sobre las particiones
        for split_name, split_imgs in splits.items():
            for img_path in split_imgs:
                # Usar un nombre único si hay colisiones (ej: prefijar con la carpeta padre)
                img_name = f"{img_path.parent.name}_{img_path.name}"
                dest_path = os.path.join(processed_dir, split_name, quality_cls, img_name)
                
                try:
                    # 1. Ejecutar Fase A: Carga y Segmentación
                    img_cropped, contour = load_and_segment_fruit(str(img_path))
                    
                    # 2. Contrato Deep Learning (Matthew): Guardar en disco la imagen procesada
                    # Redimensionamos estáticamente a 128x128 tal como espera la CNN
                    img_resized = cv2.resize(img_cropped, (128, 128), interpolation=cv2.INTER_LINEAR)
                    
                    # OpenCV trabaja con BGR por defecto. Convertimos RGB a BGR para guardar.
                    img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dest_path, img_bgr_to_save)
                    
                    # 3. Contrato ML Tradicional (Juanes): Extracción de Características
                    ml_features = get_traditional_features(img_cropped, contour)
                    
                    # Transformar a lista de Python estándar y concatenar el ID de la etiqueta de calidad (0, 1, 2)
                    label_id = QUALITY_MAP[quality_cls]
                    feature_row = ml_features.tolist() + [label_id]
                    all_tabular_features.append(feature_row)
                    
                    processed_count[split_name] += 1
                    
                except Exception as e:
                    # Robustez: Manejar archivos corruptos o fallos de segmentación
                    logger.error(f"Fallo aislado procesando imagen {img_path}: {str(e)}")
                    error_count += 1
                    
        total_processed = sum(processed_count.values())
        logger.info(f" Clase '{quality_cls}' completada. Procesadas: {total_processed}/{len(images)} (Train:{processed_count['train']} | Val:{processed_count['val']} | Test:{processed_count['test']} | Errores:{error_count})")
        
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
