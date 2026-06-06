#!/usr/bin/env python3
"""
Modulo de preparacion de datos optimizado con Multiprocesamiento para AgroVision-QC-Pipeline.

Este modulo realiza el preprocesamiento de las imagenes usando un pool de procesos
en paralelo (ProcessPoolExecutor), permitiendo procesar grandes volumenes de imagenes
(como el dataset de 10 GB) en una fraccion del tiempo original.
"""

import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Asegurar que el PYTHONPATH incluya la raiz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.helpers import setup_logger
from src.data.preprocess import load_and_segment_fruit, get_traditional_features

logger = setup_logger("AgroVision-ParallelDataset")

# Mapeo oficial de calidad
QUALITY_MAP = {
    'buena': 0,
    'media': 1,
    'mala': 2
}


def ensure_directories_exist(base_dir: str, classes: list, splits: list = ['train', 'val', 'test']):
    """Construye la estructura de carpetas de salida en data/processed."""
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)


def get_stratified_split(class_images: list, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    """Realiza una particion aleatoria estratificada a nivel de clase."""
    import random
    random.seed(42)
    shuffled_imgs = list(class_images)
    random.shuffle(shuffled_imgs)
    
    total = len(shuffled_imgs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    return {
        'train': shuffled_imgs[:train_end],
        'val': shuffled_imgs[train_end:val_end],
        'test': shuffled_imgs[val_end:]
    }


def process_single_image_worker(task):
    """Funcion de trabajo (worker) que procesa una sola imagen.

    Se ejecuta en un proceso independiente para evitar el bloqueo del GIL.
    
    Args:
        task (tuple): (img_path, dest_path, quality_cls, label_id)
        
    Returns:
        tuple: (feature_row, error_message)
    """
    img_path, dest_path, quality_cls, label_id = task
    try:
        # 1. Cargar y segmentar imagen
        img_cropped, contour = load_and_segment_fruit(str(img_path))
        
        # 2. Redimensionar y guardar imagen procesada para CNN
        img_resized = cv2.resize(img_cropped, (128, 128), interpolation=cv2.INTER_LINEAR)
        img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dest_path, img_bgr_to_save)
        
        # 3. Extraccion de caracteristicas tradicionales para ML
        ml_features = get_traditional_features(img_cropped, contour)
        feature_row = ml_features.tolist() + [label_id]
        
        return feature_row, None
    except Exception as e:
        return None, f"Error en {img_path.name}: {str(e)}"


def process_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed", max_workers: int = None, csv_path: str = "experiments/results/features_traditional_ml.csv"):
    """Orquestador que paraleliza el procesamiento del dataset usando ProcessPoolExecutor.

    Args:
        raw_dir (str): Directorio de origen de imagenes crudas.
        processed_dir (str): Directorio de destino para guardar las imagenes procesadas.
        max_workers (int): Numero maximo de nucleos de CPU a utilizar (None para automatico).
        csv_path (str): Ruta para exportar el archivo CSV de caracteristicas para ML tradicional.
    """
    start_time = time.time()
    logger.info("=== Iniciando Procesamiento del Dataset en Paralelo ===")
    
    if not os.path.exists(raw_dir):
        # Intentar con src/data/raw como contingencia
        fallback_dir = os.path.join("src", "data", "raw")
        if os.path.exists(fallback_dir):
            logger.info(f"Directorio '{raw_dir}' no encontrado. Utilizando fallback: '{fallback_dir}'")
            raw_dir = fallback_dir
        else:
            logger.error(f"El directorio crudo {raw_dir} no existe en la raíz ni en '{fallback_dir}'.")
            return

    # Usar todos los cores disponibles si no se define max_workers
    if max_workers is None:
        max_workers = os.cpu_count()
    logger.info(f"Configurando Pool con {max_workers} procesos de trabajo (CPU cores).")

    target_classes = list(QUALITY_MAP.keys())
    ensure_directories_exist(processed_dir, target_classes)
    
    # Recolectar imagenes
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images_by_quality = {cls: [] for cls in target_classes}
    
    logger.info("Escaneando imagenes crudas recursivamente...")
    for path in Path(raw_dir).rglob('*'):
        if path.suffix.lower() in valid_exts:
            path_str = str(path).lower()
            if 'buena' in path_str:
                images_by_quality['buena'].append(path)
            elif 'media' in path_str:
                images_by_quality['media'].append(path)
            elif 'mala' in path_str:
                images_by_quality['mala'].append(path)

    # Preparar lista de tareas para el Pool
    tasks = []
    total_images_scanned = 0
    
    for quality_cls, images in images_by_quality.items():
        if not images:
            continue
        total_images_scanned += len(images)
        logger.info(f"Clase '{quality_cls}': {len(images)} imagenes encontradas. Creando particiones estratificadas...")
        splits = get_stratified_split(images)
        
        for split_name, split_imgs in splits.items():
            for img_path in split_imgs:
                img_name = f"{img_path.parent.name}_{img_path.name}"
                dest_path = os.path.join(processed_dir, split_name, quality_cls, img_name)
                label_id = QUALITY_MAP[quality_cls]
                tasks.append((img_path, dest_path, quality_cls, label_id))

    if not tasks:
        logger.warning("No se encontraron imagenes para procesar en el directorio de origen.")
        return

    logger.info(f"Total de tareas de procesamiento en cola: {len(tasks)}")
    
    # Ejecucion paralela
    all_tabular_features = []
    processed_count = 0
    error_count = 0
    
    logger.info("Lanzando tareas en paralelo... Monitoreando progreso:")
    
    # Iniciar ejecucion paralela
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todas las tareas al pool
        futures = {executor.submit(process_single_image_worker, task): task for task in tasks}
        
        # Procesar los resultados a medida que se completen
        for i, future in enumerate(as_completed(futures)):
            feature_row, error_msg = future.result()
            
            if error_msg:
                error_count += 1
                # Solo logear los primeros errores para no inundar la consola
                if error_count <= 10:
                    logger.warning(error_msg)
            else:
                processed_count += 1
                all_tabular_features.append(feature_row)
            
            # Log de progreso cada 500 imagenes
            if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                logger.info(f" -> Progreso: {i + 1}/{len(tasks)} procesadas ({((i + 1)/len(tasks))*100:.1f}%) | Velocidad: {speed:.2f} imgs/seg | Transcurrido: {elapsed:.1f}s")

    # Exportar CSV de caracteristicas tradicionales
    headers = [
        'area', 'perimeter', 'aspect_ratio', 
        'h_mean', 's_mean', 'v_mean', 
        'h_std', 's_std', 'v_std', 'dark_pixel_ratio', 'label'
    ]
    
    try:
        df_features = pd.DataFrame(all_tabular_features, columns=headers)
        df_features.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Matriz tabular exportada correctamente a: '{csv_path}' con {len(df_features)} registros.")
    except Exception as e:
        logger.error(f"Error escribiendo el CSV de resultados: {str(e)}")

    total_time = time.time() - start_time
    logger.info("=== Procesamiento Finalizado ===")
    logger.info(f" Total procesadas con éxito: {processed_count}")
    logger.info(f" Total fallidas: {error_count}")
    logger.info(f" Tiempo total de ejecucion: {total_time:.2f} segundos")
    logger.info(f" Rendimiento promedio del sistema: {processed_count / total_time:.2f} imagenes por segundo")
    logger.info("=================================")


if __name__ == "__main__":
    # Por defecto corre sobre la estructura de produccion completa
    process_dataset()

