# Consideraciones Éticas, de Responsabilidad Profesional e Impacto 

Este documento presenta el análisis ético y la evaluación del impacto del sistema **AgroVision-QC-Pipeline** en su despliegue industrial, relacionándolo con los códigos profesionales de conducta de la **ACM** y del **IEEE**.

---

## 1. Tabla de Decisiones Éticas y Responsabilidad Profesional

| ID | Dilema / Decisión de Diseño | Contexto y Riesgo Identificado | Vinculación a Códigos Profesionales (ACM / IEEE) | Estrategia de Mitigación Técnica Implementada |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Privacidad de los Trabajadores en Planta** | Al instalar cámaras web sobre las bandas transportadoras, existe el riesgo de capturar los rostros de los operarios, violando su privacidad e infringiendo leyes de protección de datos personales. | **ACM 1.2 (Evitar daño), 1.6 (Respetar la privacidad).**<br>**IEEE 1.1 (Bienestar y privacidad pública).** | **Algoritmo de Segmentación y Filtrado:** El pipeline de preprocesamiento en `preprocess.py` solo opera sobre la Región de Interés (ROI) detectada mediante binarización de Otsu. Además, se restringe la captura a un plano cenital estrecho enfocado únicamente en la fruta. Si un rostro o silueta entra en el encuadre (por ejemplo, mediante clasificadores de rostros Haar Cascades), el script automáticamente descarta el cuadro (*frame drop*) o difumina el fondo por completo. |
| **02** | **Desbalanceo de Clases y Sesgo Algorítmico** | El EDA reveló que el 48.47% de las muestras pertenecen a la clase 'Media' y solo el 23.58% a la clase 'Mala'. Entrenar sin corregir este desbalanceo sesga los modelos, provocando clasificaciones erróneas que perjudican económicamente al agricultor o al distribuidor. | **ACM 1.4 (Ser justo y no discriminar).**<br>**IEEE 1.2 (Evitar sesgos y prácticas discriminatorias).** | **Balanceo Matemático de Frecuencia Inversa:** Se implementa balanceo en los scripts de entrenamiento (`train_traditional_ml.py` y `train_cnn.py`) mediante el parámetro `class_weight='balanced'` en Scikit-Learn y el tensor de ponderación (`weight`) en la pérdida `CrossEntropyLoss` de PyTorch. Adicionalmente, el split de datos se realiza de forma estrictamente **estratificada** (70% train / 15% val / 15% test). |
| **03** | **Uso Ético de Datos Públicos (Propiedad Intelectual)** | El sistema utiliza el dataset público "Fruit Quality Classification" obtenido de Kaggle. Utilizarlo para fines comerciales o académicos sin la debida atribución o violando su licenciamiento vulnera la propiedad intelectual. | **ACM 1.5 (Respetar la propiedad intelectual y créditos).**<br>**IEEE 1.4 (Evitar plagio e infracción de derechos).** | **Citación y Licenciamiento Transparente:** Se incluye una sección explícita de atribución y licencia en el `README.md` y la documentación técnica, indicando la procedencia del dataset de Kaggle y la licencia correspondiente, así como la autoría de las 30-50 imágenes adicionales recolectadas físicamente por los estudiantes. |

---

## 2. Matriz de Impacto en Cuatro Dimensiones

El despliegue de **AgroVision-QC-Pipeline** impacta de forma multidimensional en su entorno agroindustrial:

### A. Dimensión Social
- **Impacto:** Automatización de labores de selección manual de frutas. Puede provocar el desplazamiento de mano de obra no calificada en plantas empacadoras de alimentos.
- **Estrategia de Mitigación Técnica:** El sistema está diseñado para actuar como una herramienta de apoyo a la decisión en tiempo real (copiloto del operario) a través de la interfaz web en Streamlit, en lugar de un reemplazo autónomo. La interfaz muestra un indicador de confianza de predicción; si la confianza es menor a un umbral configurado ($<85\%$), el cuadro es marcado para revisión manual humana.

### B. Dimensión Económica
- **Impacto:** Incremento de la eficiencia en el control de calidad, reduciendo las pérdidas por desperdicio de frutas (pérdidas post-cosecha) y garantizando precios justos de venta según la calidad real del producto.
- **Estrategia de Mitigación Técnica:** Integración de algoritmos ligeros y eficientes de Machine Learning tradicional (Random Forest, SVM) y caching del lado del servidor Streamlit. Esto reduce drásticamente el costo de computación y permite ejecutar el sistema en hardware de bajo costo Edge AI / Raspberry Pi, disminuyendo las barreras económicas de entrada para pequeños productores agrícolas.

### C. Dimensión Ambiental
- **Impacto:** Huella de carbono derivada del entrenamiento de modelos de Deep Learning en servidores GPU de alta potencia.
- **Estrategia de Mitigación Técnica:** 
  1. Transferencia de aprendizaje (*Transfer Learning*) y detención temprana (*Early Stopping*) para minimizar el número de épocas necesarias en el entrenamiento de la CNN.
  2. Uso prioritario de modelos tabulares tradicionales de Scikit-Learn que entrenan en segundos en CPU convencional.

### D. Dimensión Global
- **Impacto:** Estandarización de normas de calidad de exportación hortofrutícola. Facilita la inserción de productores locales en mercados internacionales exigentes que requieren clasificaciones normalizadas de madurez y tamaño.
- **Estrategia de Mitigación Técnica:** Consistencia matemática rigurosa en el preprocesamiento de imágenes (segmentación basada en Otsu e interpolación bilineal a $128 \times 128$ píxeles), lo que elimina variaciones subjetivas inducidas por iluminación o cámaras de distintas regiones geográficas.
