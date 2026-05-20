# AgroVision-QC-Pipeline

Computer Vision system for automated fruit quality control using CRISP-DM. Features traditional ML and CNN models deployed end-to-end. Developed under ABET engineering competencies.

---

## Estructura del Proyecto

Esta es la organización del repositorio central del proyecto, estructurada para garantizar la reproducibilidad y el orden:

```text
📦 AgroVision-QC-Pipeline/
│
├── 📂 docs/                     # Documentación del proyecto
│   ├── 📜 README.md             # Documentación extendida
│   ├── 📜 arquitectura.md       # Detalles del modelo y diseño de arquitectura
│   ├── 📜 api.md                # Documentación de la API (si aplica)
│   └── 📜 instalacion.md        # Guía detallada de instalación y dependencias
│
├── 📂 src/                      # Código fuente principal
│   ├── 📂 data/                 # Scripts para cargar y preprocesar datos
│   │   └── preprocess.py
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
│   └── experiment_1.ipynb       # Exploración de datos y prototipado
│
├── 📂 experiments/              # Resultados de experimentos y métricas
│   ├── 📂 logs/                 # Logs de entrenamiento
│   ├── 📂 checkpoints/          # Modelos guardados y pesos de entrenamiento
│   └── 📂 results/              # Métricas, gráficas, matrices de confusión
│
├── 📂 tests/                    # Pruebas unitarias y de integración
│   └── test_models.py
│
├── 📜 requirements.txt          # Dependencias de Python
├── 📜 environment.yml           # Archivo de entorno de Conda
├── 📜 .gitignore                # Archivos ignorados por Git
├── 📜 LICENSE                   # Licencia del proyecto
└── 📜 README.md                 # Este archivo (descripción principal)
```

## Inicio Rápido

### Requisitos Previos

Asegúrate de tener instalado Python 3.10+ y/o Conda.

### Instalación con Conda

1. Crea el entorno virtual a partir de `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```
2. Activa el entorno:
   ```bash
   conda activate agrovision-qc-pipeline
   ```

### Instalación con Pip

1. Crea un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usa: venv\Scripts\activate
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Ejecución del Pipeline

Para ejecutar el pipeline principal del proyecto:

```bash
python src/main.py
```

Consulte el directorio [docs/](file:///Users/luiscadena/Documents/Universidad/semestre7/APO_III/ProyectoFinal/Proyecto/AgroVision-QC-Pipeline/docs) para obtener más información y guías de arquitectura.
