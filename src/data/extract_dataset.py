import os
import zipfile
import shutil
from pathlib import Path
from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-ExtractDataset")

# Extensiones de imagen soportadas
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def extract_and_structure_zip(project_root: str = ".", target_raw_dir: str = "data/raw") -> bool:
    """
    Busca archivos .zip en la raíz del proyecto. Si encuentra alguno, lo descomprime
    y clasifica sus imágenes en la carpeta data/raw/ según las clases:
    - buena (Good Quality_Fruits)
    - media (Regular Qualit_Fruits)
    - mala (Bad Quality_Fruits)
    
    Retorna True si encontró y procesó un archivo ZIP, False en caso contrario.
    """
    root_path = Path(project_root).resolve()
    
    # Buscar archivos .zip en la raíz y en la carpeta src/
    zip_files = [f for f in root_path.glob("*.zip") if f.is_file()]
    if not zip_files:
        zip_files = [f for f in (root_path / "src").glob("*.zip") if f.is_file()]
    
    if not zip_files:
        logger.info("No se encontraron archivos ZIP en la raíz ni en 'src/' para extraer.")
        return False
        
    zip_file = zip_files[0]
    logger.info(f"¡Archivo ZIP detectado!: '{zip_file.name}' (ubicado en '{zip_file.parent.name}'). Iniciando proceso de extracción...")
    
    # Crear carpeta temporal de extracción y carpeta destino data/raw
    temp_dir = root_path / "data" / "temp_extract"
    raw_dir = Path(target_raw_dir).resolve()
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Descomprimir el archivo ZIP en la carpeta temporal
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        logger.info(f"Descompresión completada en carpeta temporal: '{temp_dir.relative_to(root_path)}'")
        
        # Contadores para reporte final
        stats = {"buena": 0, "media": 0, "mala": 0, "omitidas": 0}
        
        # 2. Recorrer de forma recursiva buscando imágenes
        for path in temp_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                path_str = str(path).lower()
                
                # Clasificar según patrones de texto en la ruta o nombre
                target_cls = None
                if "good quality" in path_str or "good" in path_str or "buena" in path_str:
                    target_cls = "buena"
                elif "regular quality" in path_str or "regular qualit" in path_str or "regular" in path_str or "media" in path_str:
                    target_cls = "media"
                elif "bad quality" in path_str or "bad" in path_str or "mala" in path_str:
                    target_cls = "mala"
                
                if target_cls:
                    # Crear directorio de calidad correspondiente si no existe
                    dest_cls_dir = raw_dir / target_cls
                    dest_cls_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generar nombre único para la imagen (padre_nombre.ext) para evitar colisiones
                    unique_name = f"{path.parent.name}_{path.name}"
                    dest_file_path = dest_cls_dir / unique_name
                    
                    # Copiar el archivo
                    shutil.copy2(path, dest_file_path)
                    stats[target_cls] += 1
                else:
                    stats["omitidas"] += 1
                    
        total_extracted = sum(stats.values()) - stats["omitidas"]
        logger.info(
            f"Proceso de clasificación finalizado.\n"
            f"  -> Imágenes copiadas a 'data/raw/':\n"
            f"     - Buena (Good Quality):    {stats['buena']}\n"
            f"     - Media (Regular Quality):  {stats['media']}\n"
            f"     - Mala (Bad Quality):      {stats['mala']}\n"
            f"     - Omitidas (no clasificadas): {stats['omitidas']}\n"
            f"  -> Total de imágenes estructuradas: {total_extracted}"
        )
        
        # 3. Limpiar carpeta temporal
        shutil.rmtree(temp_dir)
        logger.info("Carpeta temporal limpia exitosamente.")
        
        # Opcional: renombrar o mover el ZIP original para que no se vuelva a procesar
        processed_zip = zip_file.with_name(f"processed_{zip_file.name}")
        if processed_zip.exists():
            os.remove(processed_zip)
        zip_file.rename(processed_zip)
        logger.info(f"Archivo ZIP renombrado a '{processed_zip.name}' para evitar futuras extracciones redundantes.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error crítico durante la extracción/clasificación del ZIP: {str(e)}")
        # Limpiar carpeta temporal si falla
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False

if __name__ == "__main__":
    # Test rápido de ejecución
    extract_and_structure_zip()
