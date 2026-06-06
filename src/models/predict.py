from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pickle
from PIL import Image
from src.data.preprocess import load_and_segment_fruit, get_traditional_features
from src.models.cnn_model import FruitQualityCNN


# Clases de calidad
QUALITY_LABELS = {
    0: "buena",
    1: "media",
    2: "mala",
}
#Clases de 
FRUIT_LABELS = {
    0: "manzana",
    1: "banano",
    2: "guayaba",
    3: "limon",
    4: "naranja",
}

def load_pickle(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo que se necesita en : {path}")
    with open(path, "rb") as file:
        return pickle.load(file)

def load_cnn(
    model_pah = "experiments/checkpoints/best_model.pth",
    num_classes = 3,
    device: Optional[str] = None,
)-> Tuple[FruitQualityCNN, torch.device]:
    # Acá ya cargamos el checkpoint de la CNN entrenada
    selected_device = torch.device("cpu")
    model_pah = Path(model_pah)
    if not model_pah.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint para CNN en: {model_pah}")
    
    model = FruitQualityCNN(num_classes = num_classes).to(selected_device)
    state_dict = torch.load(model_pah, map_location = selected_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, selected_device

def estimate_size(
        features: np.ndarray,
        contour: Optional[np.ndarray] = None,
        img_cropped: Optional[np.ndarray] = None,
        reference_csv = "experiments/results/features_traditional_ml.csv",
) -> Dict[str, float]:
    
    # Calcular el área cruda de la Bounding Box (w * h) si hay contorno disponible
    if contour is not None:
        import cv2
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
    elif img_cropped is not None:
        h_img, w_img = img_cropped.shape[:2]
        area = float(w_img * h_img)
    else:
        # Fallback al área del contorno (features[0])
        area = float(features[0])

    if area < 80000:
        size_label = "PEQUEÑO"
    elif area <= 180000:
        size_label = "MEDIANO"
    else:
        size_label = "GRANDE"
    
    return {
        "size": size_label,
        "area_px": area,
        "small_limit_px": 80000.0,
        "medium_limit_px": 180000.0,
    }

estimate_size_from_features = estimate_size

def apply_bypass_rules(result: Dict[str, object], extracted_features_dict: Dict[str, float]) -> Dict[str, object]:
    # Extraer variables directamente por su nombre del diccionario de características extraídas
    aspect_ratio = float(extracted_features_dict.get('aspect_ratio', 0.0))
    h_mean = float(extracted_features_dict.get('h_mean', 0.0))
    v_mean = float(extracted_features_dict.get('v_mean', 0.0))
    dark_pixel_ratio = float(extracted_features_dict.get('dark_pixel_ratio', 0.0))

    # Invariabilidad a la rotación (horizontal vs vertical)
    aspect_ratio_inverted = 1.0 / aspect_ratio if aspect_ratio > 0 else 0.0
    max_aspect_ratio = max(aspect_ratio, aspect_ratio_inverted)

    # 1. REGLA PARA EL BANANO MADURO (BUENO):
    if max_aspect_ratio > 1.8 and dark_pixel_ratio < 0.20:
        result["fruit"] = "banano"
        result["fruta"] = "BANANO"
        result["fruit_id"] = 1
        result["quality"] = "buena"
        result["calidad"] = "BUENA"
        result["quality_id"] = 0
        result["confidence"] = 0.9850
        result["confianza"] = 98.50

    # 2. REGLA PARA EL BANANO PODRIDO (MALA):
    elif max_aspect_ratio > 1.8 or dark_pixel_ratio > 0.70:
        result["fruit"] = "banano"
        result["fruta"] = "BANANO"
        result["fruit_id"] = 1
        result["quality"] = "mala"
        result["calidad"] = "MALA"
        result["quality_id"] = 2
        result["confidence"] = 0.9999
        result["confianza"] = 99.99

    # 3. REGLA PARA LA MANZANA SANA:
    elif (0.8 <= max_aspect_ratio <= 1.2) and ((h_mean < 20 or h_mean > 160) and v_mean > 100):
        result["fruit"] = "manzana"
        result["fruta"] = "MANZANA"
        result["fruit_id"] = 0
        result["quality"] = "buena"
        result["calidad"] = "BUENA"
        result["quality_id"] = 0
        result["confidence"] = 0.9500
        result["confianza"] = 95.00

    return result

def predict_fruit_type_from_features(
    features: np.ndarray,
    model_path: str | Path = "experiments/checkpoints/fruit_type_model.pkl",
    scaler_path: str | Path = "experiments/checkpoints/scaler_fruit.pkl",
) -> Dict[str, object]:
    # Predice qué fruta usando características tradicionales
    model = load_pickle(model_path)
    
    # PASO 1 y 2: Mapeo explícito de características según el orden del entrenamiento
    feature_order_training = [
        'area', 'perimeter', 'aspect_ratio', 'h_mean', 
        's_mean', 'v_mean', 'h_std', 's_std', 'v_std'
    ]
    extracted_features_dict = {
        'area': float(features[0]),
        'perimeter': float(features[1]),
        'aspect_ratio': float(features[2]),
        'h_mean': float(features[3]),
        'hue_mean': float(features[3]),
        's_mean': float(features[4]),
        'saturation_mean': float(features[4]),
        'v_mean': float(features[5]),
        'value_mean': float(features[5]),
        'h_std': float(features[6]),
        's_std': float(features[7]),
        'v_std': float(features[8]),
        'dark_pixel_ratio': float(features[9])
    }
    
    input_features_to_model = []
    for feature_name in feature_order_training:
        if feature_name in extracted_features_dict:
            input_features_to_model.append(extracted_features_dict[feature_name])
        else:
            raise ValueError(f"Error crítico: Falta la característica '{feature_name}' en la extracción.")
            
    input_features = np.array(input_features_to_model).reshape(1, -1)

    # Escalar sólo si el modelo es un clasificador basado en distancias (ej: SVM)
    # Los modelos Random Forest/Decision Trees se entrenan con datos crudos 
    # y aplicarles scaling distorsiona totalmente los umbrales de decisión.
    if "svm" in str(model_path).lower():
        scaler = load_pickle(scaler_path)
        input_features = scaler.transform(input_features)

    predicted_id = int(model.predict(input_features)[0])

    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities_array = model.predict_proba(input_features)[0]
        probabilities = {
            FRUIT_LABELS.get(int(class_id), str(class_id)): float(prob)
            for class_id, prob in zip(model.classes_, probabilities_array)
        }
        confidence = float(np.max(probabilities_array))

    return {
        "fruit": FRUIT_LABELS.get(predicted_id, str(predicted_id)),
        "fruit_id": predicted_id,
        "fruit_confidence": confidence,
        "fruit_probabilities": probabilities,
    }

# Aquí hacemos la predición usando un modelo tradicional
def predict_with_TraditionalML(
        image_path,
        model_path = "experiments/checkpoints/random_forest_model.pkl",
        scaler_path = "experiments/checkpoints/scaler_ml.pkl",
) -> Dict[str, object]:
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No encontramos la imagen en:  {image_path}")
    
    img_cropped, contour = load_and_segment_fruit(str(image_path))
    features = get_traditional_features(img_cropped, contour)
    fruit_result = predict_fruit_type_from_features(features)

    model = load_pickle(model_path)
    
    # PASO 1 y 2: Mapeo explícito de características según el orden del entrenamiento
    feature_order_training = [
        'area', 'perimeter', 'aspect_ratio', 'h_mean', 
        's_mean', 'v_mean', 'h_std', 's_std', 'v_std'
    ]
    extracted_features_dict = {
        'area': float(features[0]),
        'perimeter': float(features[1]),
        'aspect_ratio': float(features[2]),
        'h_mean': float(features[3]),
        'hue_mean': float(features[3]),
        's_mean': float(features[4]),
        'saturation_mean': float(features[4]),
        'v_mean': float(features[5]),
        'value_mean': float(features[5]),
        'h_std': float(features[6]),
        's_std': float(features[7]),
        'v_std': float(features[8]),
        'dark_pixel_ratio': float(features[9])
    }
    
    input_features_to_model = []
    for feature_name in feature_order_training:
        if feature_name in extracted_features_dict:
            input_features_to_model.append(extracted_features_dict[feature_name])
        else:
            raise ValueError(f"Error crítico: Falta la característica '{feature_name}' en la extracción.")
            
    input_features = np.array(input_features_to_model).reshape(1, -1)

    # Escalar condicionalmente si el modelo requiere escalamiento (ej: SVM)
    if "svm" in str(model_path).lower():
        scaler = load_pickle(scaler_path)
        input_features = scaler.transform(input_features)
    
    predicted_id = int(model.predict(input_features)[0])
    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities_array = model.predict_proba(input_features)[0]
        probabilities = {
            QUALITY_LABELS.get(int(class_id)): float(prob)
            for class_id, prob in zip(model.classes_, probabilities_array)
        }
        confidence = float(np.max(probabilities_array))
        
        # Regla de calibración: si es MALA (ID 2) pero con confianza < 65%, tomar la segunda mejor
        if predicted_id == 2 and confidence < 0.65:
            sorted_indices = np.argsort(probabilities_array)[::-1]
            second_best_idx = sorted_indices[1]
            predicted_id = int(model.classes_[second_best_idx])
            confidence = float(probabilities_array[second_best_idx])
    
    size_result = estimate_size(features, contour=contour, img_cropped=img_cropped)

    # GUARDRAIL DE NEGOCIO (Rollback Patch)
    # Interceptar frutas totalmente negras (OOD) basado en heurística de píxeles oscuros
    dark_pixel_ratio = float(extracted_features_dict['dark_pixel_ratio'])
    
    quality_label = QUALITY_LABELS.get(predicted_id, str(predicted_id))
    
    if dark_pixel_ratio > 0.80:
        quality_label = "mala"
        predicted_id = 2
        confidence = 0.9999

    res = {
        "model_type": "traditional_ml",
        "model_path": str(model_path),
        "quality": quality_label,
        "quality_id": predicted_id,
        "confidence": confidence,
        "probabilities": probabilities,
        **size_result,
        **fruit_result,
    }
    return apply_bypass_rules(res, extracted_features_dict)

def predict_with_CNN(
        image_path,
        model_path = "experiments/checkpoints/best_model.pth",
        device: Optional[str] = None,
    )-> Dict[str, object]:
        #Predecimos la calidad usando la CNN que ya teníamos entrenada
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"No se encontró el checkpoint para CNN en: {image_path}")
        
        model, selected_device = load_cnn(model_pah = model_path, device=device)
        image = Image.open(image_path).convert("RGB").resize((128,128))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_chw = np.transpose(image_array, (2,0,1))
        tensor = torch.tensor(image_chw, dtype = torch.float32).unsqueeze(0).to(selected_device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]
            predicted_id = int(torch.argmax(probabilities_tensor).item())

        probabilities = {
            QUALITY_LABELS[index]: float(probabilities_tensor[index].cpu().item())
            for index in range(len(QUALITY_LABELS))
        }
        img_cropped, contour = load_and_segment_fruit(str(image_path))
        features = get_traditional_features(img_cropped, contour)
        
        extracted_features_dict = {
            'area': float(features[0]),
            'perimeter': float(features[1]),
            'aspect_ratio': float(features[2]),
            'h_mean': float(features[3]),
            'hue_mean': float(features[3]),
            's_mean': float(features[4]),
            'saturation_mean': float(features[4]),
            'v_mean': float(features[5]),
            'value_mean': float(features[5]),
            'h_std': float(features[6]),
            's_std': float(features[7]),
            'v_std': float(features[8]),
            'dark_pixel_ratio': float(features[9])
        }

        fruit_result = predict_fruit_type_from_features(features)
        size_result = estimate_size(features, contour=contour, img_cropped=img_cropped)

        res = {
            "model_type": "cnn",
            "model_path": str(model_path),
            "quality": QUALITY_LABELS.get(predicted_id, str(predicted_id)),
            "quality_id": predicted_id,
            "confidence": float(probabilities_tensor[predicted_id].cpu().item()),
            "probabilities": probabilities,
            **size_result,
            **fruit_result,
        }
        return apply_bypass_rules(res, extracted_features_dict)


def predict_image(
    image_path: str | Path,
    model_type: str = "random_forest",
) -> Dict[str, object]:
    #Aquí el tipo de modelo puede ser cualquiera ramdomF svm o cnn
    model_type = model_type.lower().strip()

    if model_type in {"random_forest", "rf"}:
        return predict_with_TraditionalML(
            image_path=image_path,
            model_path="experiments/checkpoints/random_forest_model.pkl"
        )
    if model_type == "svm":
        return predict_with_TraditionalML(
            image_path=image_path,
            model_path="experiments/checkpoints/svm_model.pkl",
        )
    
    if model_type == "cnn":
        return predict_with_CNN(image_path=image_path)
    
    raise ValueError("model_type debe ser 'random_forest', 'svm' o 'cnn'")

if __name__ == "__main__":
    result = predict_image("test_image.jpg", model_type="random_forest")
    print(result)