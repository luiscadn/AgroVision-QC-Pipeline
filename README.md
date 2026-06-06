# AgroVision-QC-Pipeline

Computer Vision system for automated fruit quality control using CRISP-DM. Features traditional ML and CNN models deployed end-to-end. Developed under ABET engineering competencies.

---

## Estructura del Proyecto

Esta es la organización del repositorio central del proyecto, estructurada para garantizar la reproducibilidad y el orden:

```text
📦 AgroVision-QC-Pipeline/
│
├── 📂 data/                     # Datos del proyecto (excluidos de Git)
│   ├── 📂 raw/                  # Imágenes originales descargadas (buena/media/mala)
│   ├── 📂 processed/            # Imágenes preprocesadas para entrenamiento
│   ├── 📂 processed_fruit/      # Imágenes organizadas por tipo de fruta
│   └── 📂 external/             # Recursos externos adicionales
│
├── 📂 scripts/                  # Scripts utilitarios del proyecto
│   └── download_kaggle_dataset.py  # ⬇️ Descarga automática del dataset de Kaggle
│
├── 📂 docs/                     # Documentación del proyecto
│   ├── 📜 README.md             # Documentación extendida
│   ├── 📜 arquitectura.md       # Detalles del modelo y diseño de arquitectura
│   ├── 📜 api.md                # Documentación de la API (si aplica)
│   ├── 📜 instalacion.md        # Guía detallada de instalación y dependencias
│   └── 📜 dataset_setup.md      # 🔑 Guía de credenciales Kaggle y descarga de datos
│
├── 📂 src/                      # Código fuente principal
│   ├── 📂 data/                 # Scripts para cargar y preprocesar datos
│   │   ├── make_dataset.py      # Orquestador del pipeline de datos (incluye descarga)
│   │   ├── extract_dataset.py   # Extracción de ZIPs locales
│   │   └── preprocess.py        # Preprocesamiento e ingeniería de características
│   ├── 📂 models/               # Definición de arquitecturas de modelos (ML y CNN)
│   │   └── my_model.py
│   ├── 📂 training/             # Scripts de entrenamiento y ajuste de hiperparámetros
│   │   └── train.py
│   ├── 📂 evaluation/           # Scripts de validación y pruebas de desempeño
│   │   └── evaluate.py
│   ├── 📂 utils/                # Funciones auxiliares
│   │   └── helpers.py
│   └── main.py                  # Punto de entrada principal para el pipeline
│
├── 📂 notebooks/                # Jupyter Notebooks para experimentación
│   └── experiment_1.ipynb
│
├── 📂 experiments/              # Resultados de experimentos y métricas
│   ├── 📂 logs/
│   ├── 📂 checkpoints/
│   └── 📂 results/
│
├── 📂 tests/                    # Pruebas unitarias y de integración
│   └── test_models.py
│
├── 📜 requirements.txt          # Dependencias de Python
├── 📜 environment.yml           # Archivo de entorno de Conda
├── 📜 .gitignore                # Archivos ignorados por Git
└── 📜 README.md                 # Este archivo (descripción principal)
```

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.10+
- Cuenta en [Kaggle](https://www.kaggle.com) con API key generada
- Git

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/AgroVision-QC-Pipeline.git
cd AgroVision-QC-Pipeline
```

### 2. Crear y activar el entorno virtual (venv)

```bash
# Crear el entorno
python -m venv venv

# Activar — Windows PowerShell
venv\Scripts\Activate.ps1

# Activar — Windows CMD
venv\Scripts\activate.bat

# Activar — Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup (Kaggle API)

El proyecto usa **dos datasets** de frutas combinados:
- Dataset principal del equipo (imágenes en `data/raw/`)
- **Kaggle — Fruit Quality Classification** (`ryandpark/fruit-quality-classification`)

### Paso 1 — Obtener tu API key de Kaggle

1. Inicia sesión en [https://www.kaggle.com](https://www.kaggle.com).
2. Ve a tu **perfil** (esquina superior derecha) → **Settings**.
3. Desplázate hasta la sección **API** y haz clic en **"Create New Token"**.
4. Se descargará automáticamente el archivo `kaggle.json` con el siguiente formato:
   ```json
   {"username": "tu_usuario_kaggle", "key": "tu_api_key_kaggle"}
   ```

### Paso 2 — Configurar las credenciales

Tienes dos opciones (usa la que prefieras):

#### Opción A — Archivo `.env` en la raíz del proyecto *(recomendada para equipos)*

Crea un archivo `.env` en la raíz del repositorio con el siguiente contenido:

```env
KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_api_key_kaggle
```

> ⚠️ El archivo `.env` está incluido en `.gitignore`. **Nunca lo subas al repositorio.**

#### Opción B — Archivo `kaggle.json` en tu carpeta de usuario *(estándar oficial)*

| Sistema Operativo | Ruta destino |
|---|---|
| Windows | `C:\Users\<TU_USUARIO>\.kaggle\kaggle.json` |
| Linux / macOS | `~/.kaggle/kaggle.json` |

```bash
# Linux / macOS — crear la carpeta y asignar permisos seguros
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

```powershell
# Windows PowerShell — crear la carpeta y mover el archivo
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

### Paso 3 — Descargar el dataset

#### Opción automática (recomendada) — integrado en el pipeline de datos

Al ejecutar el pipeline principal, la descarga se activa automáticamente si `data/raw/` está vacío:

```bash
python src/data/make_dataset.py
```

#### Opción manual — ejecutar sólo la descarga

```bash
python scripts/download_kaggle_dataset.py
```

Para forzar una nueva descarga aunque ya existan datos:

```bash
python scripts/download_kaggle_dataset.py --force
```

Consulta la guía completa en [docs/dataset_setup.md](docs/dataset_setup.md).

---

## 🛠️ Ejecución del Pipeline

Para ejecutar el pipeline principal del proyecto:

```bash
python src/main.py
```

Consulte el directorio [docs/](docs/) para obtener más información y guías de arquitectura.

