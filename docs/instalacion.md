# Guía de Instalación y Configuración

Sigue estos pasos detallados para instalar todas las dependencias y configurar tu entorno para el proyecto **AgroVision-QC-Pipeline**.

## Requisitos de Software

- **Python 3.10** o superior
- **Conda** (opcional, pero recomendado para gestión de entornos aislados)
- **Git**

## Configuración del Entorno Virtual

### Opción 1: Conda (Recomendado)

Crea un entorno virtual usando Conda para aislar todas las dependencias específicas de procesamiento de imágenes y Machine Learning:

```bash
# Crear el entorno a partir de environment.yml
conda env create -f environment.yml

# Activar el entorno creado
conda activate agrovision-qc-pipeline
```

### Opción 2: Pip y Virtualenv (Python Estándar)

Si no usas Conda, puedes configurar un entorno virtual estándar de Python:

```bash
# Crear entorno virtual en la carpeta venv
python -m venv venv

# Activar en Linux/macOS
source venv/bin/activate

# Activar en Windows
venv\Scripts\activate

# Instalar dependencias requeridas
pip install --upgrade pip
pip install -r requirements.txt
```

## Solución de Problemas Comunes

- **Error al compilar dependencias C**: Asegúrate de tener las herramientas de compilación de C++ instaladas en tu sistema (Xcode Command Line Tools en macOS o Build Tools para Visual Studio en Windows).
- **Problemas con TensorFlow / GPU**: Verifica que los controladores de NVIDIA y CUDA estén configurados correctamente si deseas utilizar la aceleración por hardware.
