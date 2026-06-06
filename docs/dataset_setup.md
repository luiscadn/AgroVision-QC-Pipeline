# Guía de Dataset y Credenciales Kaggle

Esta guía explica en detalle cómo configurar la API de Kaggle para descargar
el dataset **Fruit Quality Classification** y cómo está integrado al pipeline
del proyecto AgroVision-QC-Pipeline.

---

## Índice

1. [¿Por qué usamos Kaggle API?](#por-qué-usamos-kaggle-api)
2. [Datasets utilizados](#datasets-utilizados)
3. [Crear tu cuenta y API key en Kaggle](#crear-tu-cuenta-y-api-key-en-kaggle)
4. [Configurar credenciales (Opción A — `.env`)](#opción-a--archivo-env)
5. [Configurar credenciales (Opción B — `kaggle.json`)](#opción-b--archivo-kaggle-json)
6. [Estructura de carpetas de datos](#estructura-de-carpetas-de-datos)
7. [Descargar el dataset](#descargar-el-dataset)
8. [Ejecutar el pipeline completo](#ejecutar-el-pipeline-completo)
9. [Preguntas frecuentes](#preguntas-frecuentes)

---

## ¿Por qué usamos Kaggle API?

Almacenar imágenes directamente en el repositorio de Git no es viable porque:

- Los archivos binarios grandes degradan el rendimiento de Git.
- Los datos ocupan varios GB y no deben compartirse por correo o USB.
- Cada integrante del equipo necesita los **mismos datos y la misma organización**.

La API de Kaggle resuelve esto: cualquier persona que clone el repositorio puede
ejecutar **un solo comando** para obtener los datos exactos sin depender de archivos
en el computador de un solo integrante.

---

## Datasets utilizados

| Dataset | Fuente | Carpeta destino |
|---|---|---|
| Dataset principal del equipo | Archivos locales / ZIP | `data/raw/` |
| **Fruit Quality Classification** | [Kaggle — ryandpark](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) | `data/raw/` |

Ambos datasets se fusionan en `data/raw/` bajo la estructura:

```
data/raw/
├── buena/   ← imágenes de buena calidad
├── media/   ← imágenes de calidad media
└── mala/    ← imágenes de mala calidad
```

---

## Crear tu cuenta y API key en Kaggle

> [!IMPORTANT]
> Cada integrante del equipo debe crear su propia API key. **No compartas tu key con nadie.**

### Pasos

1. Ve a [https://www.kaggle.com](https://www.kaggle.com) y crea una cuenta (es gratuita).
2. Inicia sesión y haz clic en tu **foto de perfil** (esquina superior derecha).
3. Selecciona **Settings**.
4. Desplázate hasta la sección **API**.
5. Haz clic en **"Create New Token"**.
6. Se descargará automáticamente un archivo llamado `kaggle.json` con este contenido:

```json
{
  "username": "tu_nombre_de_usuario",
  "key": "tu_api_key_de_32_caracteres"
}
```

---

## Opción A — Archivo `.env`

Esta es la opción **recomendada para equipos** porque no requiere mover archivos
fuera del proyecto y es más fácil de recordar.

### Pasos

1. En la raíz del proyecto (`AgroVision-QC-Pipeline/`), crea un archivo llamado **`.env`**.
2. Ábrelo con cualquier editor de texto y escribe:

```env
KAGGLE_USERNAME=tu_nombre_de_usuario
KAGGLE_KEY=tu_api_key_de_32_caracteres
```

3. Guarda el archivo.

> [!CAUTION]
> El archivo `.env` ya está en `.gitignore`. **Nunca ejecutes `git add .env`**
> ni lo subas al repositorio. Contiene información privada.

---

## Opción B — Archivo `kaggle.json`

Esta es la forma estándar oficial de Kaggle. El archivo va en una carpeta oculta
dentro de tu carpeta de usuario del sistema operativo.

### Windows

```powershell
# 1. Crear la carpeta .kaggle si no existe
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"

# 2. Mover el kaggle.json descargado a esa carpeta
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

La ruta final debe ser:
```
C:\Users\<TU_USUARIO>\.kaggle\kaggle.json
```

### Linux / macOS

```bash
# 1. Crear la carpeta .kaggle
mkdir -p ~/.kaggle

# 2. Mover el archivo
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# 3. Asignar permisos seguros (requerido por la librería kaggle)
chmod 600 ~/.kaggle/kaggle.json
```

---

## Estructura de carpetas de datos

```
data/
├── raw/            ← Datos originales (NO modificar manualmente)
│   ├── buena/
│   ├── media/
│   └── mala/
├── processed/      ← Imágenes procesadas por calidad (train/val/test)
│   ├── train/
│   ├── val/
│   └── test/
├── processed_fruit/ ← Imágenes procesadas por tipo de fruta
│   ├── train/
│   ├── val/
│   └── test/
└── external/       ← Recursos externos adicionales (modelos, lookup tables)
    └── README.md
```

> [!NOTE]
> Las carpetas `raw/`, `processed/` y `processed_fruit/` están en `.gitignore`.
> Solo `external/README.md` se rastrea en Git para documentar su propósito.

---

## Descargar el dataset

### Automáticamente (al correr el pipeline)

```bash
# Con el entorno virtual activado:
python src/data/make_dataset.py
```

Si `data/raw/` está vacío, el pipeline descarga el dataset de Kaggle antes
de comenzar el procesamiento.

### Manualmente (solo descarga)

```bash
python scripts/download_kaggle_dataset.py
```

### Forzar nueva descarga (aunque ya existan datos)

```bash
python scripts/download_kaggle_dataset.py --force
```

---

## Ejecutar el pipeline completo

Una vez configuradas las credenciales, el flujo completo desde cero es:

```bash
# 1. Activar entorno virtual
venv\Scripts\Activate.ps1        # Windows PowerShell
# o
source venv/bin/activate          # Linux/macOS

# 2. Instalar dependencias (solo la primera vez)
pip install -r requirements.txt

# 3. Ejecutar pipeline de datos (descarga + procesamiento)
python src/data/make_dataset.py

# 4. Ejecutar entrenamiento
python src/main.py
```

---

## Preguntas frecuentes

**P: ¿Qué pasa si tengo el `.env` Y el `kaggle.json`?**
El script da prioridad al `.env`. Si las variables `KAGGLE_USERNAME` y `KAGGLE_KEY`
están definidas, se usan esas. De lo contrario, busca `~/.kaggle/kaggle.json`.

---

**P: Error `401 - Unauthorized`**
Significa que la API key es incorrecta o está caducada. Ve a Kaggle → Settings → API
y genera un nuevo token. Actualiza tu `.env` o `kaggle.json` con la nueva key.

---

**P: Error `403 - Forbidden`**
Debes aceptar las reglas del dataset en Kaggle antes de poder descargarlo.
Ve a la [página del dataset](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification),
desplázate hasta el pie de página y acepta las condiciones de uso.

---

**P: ¿Por qué no se sube `data/raw/` a Git?**
Las imágenes ocupan varios GB, lo que haría el repositorio extremadamente lento.
Git está diseñado para código fuente, no para grandes archivos binarios.

---

**P: Un compañero ya tiene los datos, ¿puedo copiarlos directamente?**
Puedes, pero asegúrate de respetar la estructura de carpetas:
`data/raw/buena/`, `data/raw/media/`, `data/raw/mala/`.
De lo contrario, `make_dataset.py` no los encontrará.
