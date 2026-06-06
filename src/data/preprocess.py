import cv2
import numpy as np
import torch
from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-Data")

def load_and_segment_fruit(image_path: str):
    """
    Carga la imagen desde el disco y aplica técnicas de visión por computadora
    para segmentar y aislar la fruta del fondo. Esto es crucial en la Fase 3 
    de CRISP-DM para reducir el ruido o fondo antes de entrenar modelos.

    Args:
        image_path (str): Ruta absoluta o relativa a la imagen.
        
    Returns:
        tuple: (img_cropped, contour) donde img_cropped es la imagen recortada 
               (NumPy array RGB) y contour es el contorno de mayor área.
               Si no se detectan contornos, devuelve (img_rgb, None).
    """
    logger.info(f"Cargando imagen para segmentación: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        error_msg = f"No se pudo cargar la imagen {image_path}. Verifica que la ruta exista y el archivo no esté corrupto."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    # Convertir a RGB para que los canales de color sean interpretables estándarmente
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Conversión a escala de grises para el análisis de intensidad y umbralización
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Suavizado gaussiano para eliminar ruido de alta frecuencia (smoothness)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Umbralización (Thresholding) utilizando el método de Otsu para separar 
    # automáticamente el primer plano (fruta) del fondo (asumiendo distribución bimodal).
    # Usamos THRESH_BINARY_INV asumiendo que el fondo es claro y la fruta más oscura
    # (muy común en control de calidad con cajas de luz), u Otsu ajustará el límite óptimo.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Operaciones morfológicas (Clausura) para rellenar huecos internos del objeto
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 5. Extracción de contornos topológicos (límites de los objetos segmentados)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning(f"No se detectaron contornos en la imagen {image_path}. Retornando imagen completa sin recortes.")
        return img_rgb, None
        
    # Identificar el contorno con la mayor área, asumiendo empíricamente que la fruta es el objeto principal
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcular la caja delimitadora ortogonal (Bounding Box)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Recortar la región de interés (ROI) de la imagen original en formato RGB
    img_cropped = img_rgb[y:y+h, x:x+w]
    
    logger.info("Fruta segmentada y aislada correctamente.")
    return img_cropped, largest_contour

def get_cnn_tensor(img_cropped: np.ndarray) -> torch.Tensor:
    """
    Toma la imagen recortada y aplica las transformaciones de álgebra matricial
    requeridas para alimentar la Red Neuronal Convolucional (PyTorch).
    
    Args:
        img_cropped (np.ndarray): Imagen recortada en formato HWC y colores RGB.
        
    Returns:
        torch.Tensor: Tensor 3D en formato CHW con dimensiones (3, 128, 128) 
                      y tipo flotante de 32 bits, valores escalados a [0.0, 1.0].
    """
    logger.info("Convirtiendo ROI a Tensor para Deep Learning (CNN).")
    
    # 1. Redimensionamiento espacial estricto (128x128) garantizando el contrato de entrada del modelo CNN.
    # Usamos interpolación bilineal para mantener gradientes suaves de píxeles.
    img_resized = cv2.resize(img_cropped, (128, 128), interpolation=cv2.INTER_LINEAR)
    
    # 2. Normalización Min-Max al rango flotante [0.0, 1.0]. 
    # Fundamental matemáticamente para evitar explosión/desvanecimiento de gradientes
    # en la retropropagación (Backpropagation) y agilizar la convergencia convexa.
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 3. Transposición de ejes: De HWC (Alto, Ancho, Canales) a CHW (Canales, Alto, Ancho).
    # PyTorch requiere el canal de color en el eje 0 (dimensión externa) para realizar 
    # la multiplicación de tensores optimizada en sus capas nn.Conv2d de CUDA.
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # 4. Construcción del objeto Tensor
    cnn_tensor = torch.tensor(img_chw, dtype=torch.float32)
    
    return cnn_tensor

def get_traditional_features(img_cropped: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Extrae un Feature Vector analítico basado en heurísticas matemáticas de forma y color,
    creando un contrato estandarizado para clasificadores tabulares de Machine Learning (ej. Random Forest).
    
    Args:
        img_cropped (np.ndarray): Imagen recortada en formato RGB.
        contour (np.ndarray): Arreglo de coordenadas del contorno de la fruta.
        
    Returns:
        np.ndarray: Vector 1D (unidimensional) de características tipo float32.
    """
    logger.info("Calculando matriz de características (Features) clásicas.")
    features = []
    
    # ----- CARACTERÍSTICAS GEOMÉTRICAS (TAMAÑO/FORMA) -----
    if contour is not None:
        # Área topológica real basada en el número de píxeles encerrados (momento cero del contorno)
        area = float(cv2.contourArea(contour))
        
        # Perímetro lineal (longitud de arco de los límites discretos)
        perimeter = float(cv2.arcLength(contour, closed=True))
        
        # Proporción geométrica rectangular (Aspect Ratio)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0.0
    else:
        # Fallback en caso de que la segmentación fallara (caja completa)
        h_img, w_img = img_cropped.shape[:2]
        area = float(w_img * h_img)
        perimeter = float(2 * (w_img + h_img))
        aspect_ratio = float(w_img) / h_img if h_img != 0 else 0.0
        
    features.extend([area, perimeter, aspect_ratio])
    
    # ----- CARACTERÍSTICAS COLORIMÉTRICAS (CALIDAD/MADUREZ) -----
    # Transformación del espacio de color RGB -> HSV. 
    # Esto desacopla la cromaticidad de la luminancia, permitiendo aislar el "color puro"
    # independientemente de las condiciones de luz/sombras en la captura de la cámara.
    img_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
    
    # Descomposición en canales: H (Hue/Matiz), S (Saturation/Saturación), V (Value/Brillo)
    h_channel, s_channel, v_channel = cv2.split(img_hsv)
    
    # Medida de tendencia central (Media Matemática):
    # - H_mean define si la fruta está verde, amarilla, etc. (Madurez general)
    # - S_mean define qué tan vibrante o "limpio" es el color general.
    h_mean, s_mean, v_mean = np.mean(h_channel), np.mean(s_channel), np.mean(v_channel)
    
    # Medida de dispersión (Desviación Estándar):
    # Refleja la varianza superficial. Un alto h_std o s_std indica heterogeneidad.
    # Físicamente modela la presencia de texturas anómalas: manchas oscuras, 
    # daño en el tejido o áreas podridas que contrastan fuerte con la piel sana.
    h_std, s_std, v_std = np.std(h_channel), np.std(s_channel), np.std(v_channel)
    
    features.extend([h_mean, s_mean, v_mean, h_std, s_std, v_std])
    
    # ----- CARACTERÍSTICAS ESPECÍFICAS DE PODREDUMBRE -----
    # Para solucionar el overfitting de frutas que son completamente negras (donde la varianza es baja),
    # calculamos el porcentaje absoluto de píxeles con bajo brillo (V < 50) que son parte de la fruta.
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        shifted_contour = contour - [x, y]
        mask = np.zeros(img_cropped.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
        
        # Extraer canales V sólo para la fruta
        fruit_v = v_channel[mask == 255]
        dark_pixels = np.sum(fruit_v < 60) # Umbral de oscuridad empírico
        dark_pixel_ratio = float(dark_pixels) / max(len(fruit_v), 1)
    else:
        # Fallback
        dark_pixels = np.sum(v_channel < 60)
        dark_pixel_ratio = float(dark_pixels) / max(v_channel.size, 1)
        
    features.append(dark_pixel_ratio)
    
    # Serialización en tensor unidimensional NumPy (dtype=float32 común en ML)
    feature_vector = np.array(features, dtype=np.float32)
    
    return feature_vector

def pipeline_extractor_maestro(image_path: str) -> tuple:
    """
    Patrón orquestador principal del pipeline de preprocesamiento de datos (CRISP-DM - Fase 3).
    Toma los datos crudos y emite dos salidas divergentes cumpliendo el contrato
    con ambos frentes de desarrollo (Deep Learning y Machine Learning Tradicional).
    
    Args:
        image_path (str): Ruta absoluta o relativa al sistema de archivos del sensor/imagen.
        
    Returns:
        tuple: Un par que contiene:
               1. (torch.Tensor) Tensor CNN normalizado listo para el inferenciador de PyTorch.
               2. (np.ndarray) Vector estructurado 1D listo para modelos de Scikit-Learn.
    """
    logger.info(f"== INICIANDO PIPELINE MAESTRO == | Archivo: {image_path}")
    
    # Fase A: Segmentación Morfológica
    img_cropped, contour = load_and_segment_fruit(image_path)
    
    # Fase B: Adaptador de Interfaz CNN (Matthew)
    cnn_tensor = get_cnn_tensor(img_cropped)
    
    # Fase C: Extractor Analítico (Juanes)
    ml_feature_vector = get_traditional_features(img_cropped, contour)
    
    logger.info("== PIPELINE MAESTRO FINALIZADO EXITOSAMENTE ==")
    
    return cnn_tensor, ml_feature_vector
