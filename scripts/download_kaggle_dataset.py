"""
download_kaggle_dataset.py
==========================
Descarga automáticamente el dataset de Kaggle:
  ryandpark/fruit-quality-classification

Uso directo:
    python scripts/download_kaggle_dataset.py

También se invoca desde src/data/make_dataset.py cuando la carpeta
data/raw/ está vacía o se fuerza la descarga con --force.


Cómo obtener tu API key
-----------------------
1. Inicia sesión en https://www.kaggle.com
2. Ve a tu perfil → Settings → API → "Create New Token"
3. Descarga el archivo kaggle.json generado.
4. Colócalo en ~/.kaggle/ o copia sus valores a tu .env local.
"""

import os
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Cargar variables de entorno desde .env si existe (Opción A)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # Busca .env en la raíz del proyecto (un nivel arriba de scripts/)
    _project_root = Path(__file__).resolve().parents[1]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        print(f"[INFO] Variables de entorno cargadas desde: {_env_file}")
except ImportError:
    # python-dotenv es opcional; si no está instalado se omite silenciosamente
    pass


# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------
KAGGLE_DATASET_SLUG = "ryandpark/fruit-quality-classification"

# Directorios destino (relativos a la raíz del proyecto)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_credentials() -> None:
    """
    Valida que las credenciales de Kaggle estén disponibles antes de intentar
    la descarga. Lanza EnvironmentError con instrucciones claras si no lo están.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not (username and key) and not kaggle_json.exists():
        raise EnvironmentError(
            "\n\n"
            "   No se encontraron credenciales de la API de Kaggle.\n\n"
            "  Tienes dos opciones para configurarlas:\n\n"
            "  ── Opción A (recomendada): archivo .env en la raíz del proyecto ──\n"
            "     Crea el archivo .env con el siguiente contenido:\n"
            "         KAGGLE_USERNAME=tu_usuario_kaggle\n"
            "         KAGGLE_KEY=tu_api_key_kaggle\n\n"
            "  ── Opción B: archivo kaggle.json ──\n"
            "     Coloca el archivo en:\n"
            "         Windows → C:/Users/<TU_USUARIO>/.kaggle/kaggle.json\n"
            "         Linux/Mac → ~/.kaggle/kaggle.json\n"
            "     Contenido:\n"
            '         {"username":"tu_usuario","key":"tu_api_key"}\n\n'
            "  Para obtener tu API key:\n"
            "     1. Inicia sesión en https://www.kaggle.com\n"
            "     2. Ve a Perfil → Settings → API → 'Create New Token'\n"
            "     3. Descarga el kaggle.json generado.\n"
        )


def _raw_dir_has_images(raw_dir: Path) -> bool:
    """Devuelve True si ya existen imágenes en data/raw/ (cualquier subdirectorio)."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return any(
        f.suffix.lower() in valid_exts
        for f in raw_dir.rglob("*")
        if f.is_file()
    )


# ---------------------------------------------------------------------------
# Función principal de descarga
# ---------------------------------------------------------------------------

def download_dataset(force: bool = False) -> bool:
    """
    Descarga y descomprime el dataset de Kaggle en data/raw/.

    Parámetros
    ----------
    force : bool
        Si True, descarga incluso cuando ya existen imágenes en data/raw/.

    Retorna
    -------
    bool
        True si la descarga se realizó, False si se omitió (datos ya presentes).
    """
    # 1. Verificar credenciales
    _check_credentials()

    # 2. Importar kaggle API (falla con mensaje claro si no está instalada)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "\n    La librería 'kaggle' no está instalada.\n"
            "  Instálala con:\n"
            "      pip install kaggle\n"
        )

    # 3. Crear directorios si no existen
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Comprobar si ya hay datos descargados
    if not force and _raw_dir_has_images(RAW_DIR):
        print(
            f"[INFO] Ya existen imágenes en '{RAW_DIR.relative_to(PROJECT_ROOT)}'.\n"
            "       Omitiendo descarga. Usa --force para forzar una nueva descarga."
        )
        return False

    # 5. Autenticar y descargar
    print(f"[INFO] Autenticando con la API de Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Descargando dataset: '{KAGGLE_DATASET_SLUG}'")
    print(f"[INFO] Destino: {RAW_DIR}")

    api.dataset_download_files(
        dataset=KAGGLE_DATASET_SLUG,
        path=str(RAW_DIR),
        unzip=True,
        quiet=False,
    )

    print(f"\n[✅] Dataset descargado y descomprimido exitosamente en: {RAW_DIR}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga el dataset Fruit Quality Classification desde Kaggle."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la descarga aunque ya existan datos en data/raw/.",
    )
    args = parser.parse_args()

    try:
        downloaded = download_dataset(force=args.force)
        if downloaded:
            print("\n[INFO] Próximo paso sugerido:")
            print("       python src/data/make_dataset.py")
    except (EnvironmentError, ImportError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Fallo inesperado durante la descarga: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
