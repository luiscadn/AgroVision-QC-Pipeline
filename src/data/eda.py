#!/usr/bin/env python3
"""
Modulo para el Analisis Exploratorio de Datos EDA en AgroVision-QC-Pipeline.

Este script escanea de forma recursiva y dinamica el directorio de datos crudos,
calcula estadisticas descriptivas de distribucion y balance de clases de calidad,
y genera visualizaciones de alta resolucion para documentacion de calidad ABET.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Asegurar que el directorio src esta en el PYTHONPATH del sistema
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-EDA")

# Extensiones de imagen soportadas por la biblioteca de vision OpenCV/PIL
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def run_dataset_eda(raw_dir_path: str = "data/raw", output_plot_path: str = "docs/visualizations/class_distribution.png"):
    """Realiza el escaneo del dataset crudo y genera el reporte visual de distribucion.

    Args:
        raw_dir_path (str): Ruta relativa o absoluta al directorio de datos crudos.
        output_plot_path (str): Ruta de destino para guardar el grafico de barras de distribucion.

    Returns:
        dict: Diccionario con los conteos absolutos por clase identificada.
    """
    logger.info("=== Iniciando Analisis Exploratorio de Datos (EDA) ===")
    
    # Resolver la ruta absoluta del directorio
    raw_path = Path(raw_dir_path).resolve()
    
    # Si la ruta data/raw en la raiz no existe, intentar con src/data/raw como contingencia
    if not raw_path.exists():
        fallback_path = Path(PROJECT_ROOT) / "src" / "data" / "raw"
        if fallback_path.exists():
            logger.info(f"Directorio '{raw_path}' no encontrado. Utilizando fallback: '{fallback_path}'")
            raw_path = fallback_path
        else:
            logger.warning(
                f"El directorio de origen '{raw_dir_path}' no existe en la raiz del proyecto ni en 'src/data/raw/'. "
                "Crea la carpeta e introduce imagenes crudas para ejecutar un analisis real."
            )
            # Retornar vacio controlado para no romper ejecuciones de prueba automatizadas
            return {}

    # Estructura para registrar los conteos
    class_stats = {}
    total_images = 0
    
    logger.info(f"Escaneando directorio de datos crudos: '{raw_path}'")
    
    # 1. Escaneo dinamico tolerante a fallos con os.walk
    try:
        for root, dirs, files in os.walk(raw_path):
            root_path = Path(root)
            quality_cls = None
            
            # Determinar a que clase macro ('buena', 'media', 'mala') pertenece esta ruta
            for parent in [root_path] + list(root_path.parents):
                parent_name = parent.name.lower()
                if parent_name in {'buena', 'media', 'mala'}:
                    quality_cls = parent_name
                    break
            
            if quality_cls:
                for file in files:
                    file_path = root_path / file
                    if file_path.suffix.lower() in VALID_EXTENSIONS:
                        class_stats[quality_cls] = class_stats.get(quality_cls, 0) + 1
                        total_images += 1
    except Exception as e:
        logger.error(f"Error critico durante la lectura de archivos del dataset: {str(e)}")
        sys.exit(1)

    # Validar si se encontraron imagenes en el escaneo
    if total_images == 0:
        logger.warning(
            "El escaneo finalizo exitosamente pero no se encontraron imagenes validas "
            "(.jpg, .jpeg, .png, .bmp, .tif, .tiff) clasificadas en subcarpetas de calidad."
        )
        return {}

    # 2. Computar e Imprimir Logs Corporativos / Analisis descriptivo
    logger.info("--- Reporte de Distribucion de Calidad ---")
    logger.info(f"Total de imagenes validas detectadas: {total_images}")
    
    summary_data = []
    for cls, count in class_stats.items():
        percentage = (count / total_images) * 100
        logger.info(f" -> Clase '{cls.upper()}': {count} muestras ({percentage:.2f}%)")
        summary_data.append({"Clase": cls.capitalize(), "Muestras": count, "Porcentaje": percentage})

    # Convertir a DataFrame para visualizacion con Seaborn
    df_summary = pd.DataFrame(summary_data)
    
    # 3. Generar y Exportar Visualizaciones bajo criterios ABET (Countplot/Barplot)
    try:
        # Configurar tema visual de alta calidad
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        
        # Paleta viridis y grafico de barras
        ax = sns.barplot(
            x="Clase", 
            y="Muestras", 
            data=df_summary, 
            palette="viridis",
            hue="Clase",
            legend=False
        )
        
        # Enriquecer anotaciones y etiquetas
        plt.title("Distribucion e Imbalance de Clases de Calidad de Frutas\n(AgroVision QC)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Clasificacion de Calidad", fontsize=12, fontweight='semibold')
        plt.ylabel("Cantidad de Muestras", fontsize=12, fontweight='semibold')
        
        # Agregar etiquetas de conteo sobre cada barra
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=11, fontweight='bold'
            )

        # Crear carpeta de salida si no existe
        plot_path = Path(output_plot_path).resolve()
        os.makedirs(plot_path.parent, exist_ok=True)
        
        # Guardar en alta resolucion (300 DPI) para reporte academico
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logger.info(f"Grafico de distribucion exportado con éxito a: '{plot_path}'")
        
    except Exception as e:
        logger.error(f"Error generando o guardando la visualizacion del EDA: {str(e)}")

    logger.info("=== Analisis Exploratorio de Datos (EDA) Finalizado ===")
    return class_stats


if __name__ == "__main__":
    # Iniciar ejecucion por defecto al correr como script
    run_dataset_eda()
