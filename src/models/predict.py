from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pickle
import cv2
import streamlit as st
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
    5: "granada",
}

@st.cache_resource
def load_pickle(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo que se necesita en : {path}")
    with open(path, "rb") as file:
        return pickle.load(file)

@st.cache_resource
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
        reference_csv = "experiments/results/features_traditional_ml.csv",
) -> Dict[str, float]:
    
    area = float(features[0])
    reference_csv = Path(reference_csv)

    if reference_csv.exists():
        df = pd.read_csv(reference_csv)
        if "area" in df.columns and len(df) >= 3:
            small_limit = float(df["area"].quantile(0.33))
            medium_limit = float(df["area"].quantile(0.66))
        else:
            small_limit, medium_limit = 8_000.0, 20_000.0
    else:
        small_limit, medium_limit = 8_000.0, 20_000.0

    if area <= small_limit:
        size_label = "pequeño"
    elif area <= medium_limit:
        size_label = "mediano"
    else:
        size_label = "grande"
    
    return {
        "size": size_label,
        "area_px": area,
        "small_limit_px": small_limit,
        "medium_limit_px": medium_limit,
    }

def predict_fruit_type_from_features(
    features: np.ndarray,
    model_path: str | Path = "experiments/checkpoints/fruit_type_model.pkl",
    scaler_path: str | Path = "experiments/checkpoints/scaler_ml.pkl",
) -> Dict[str, object]:
    #Predice qué fruta usando características tradicionales

    model = load_pickle(model_path)

    input_features = features.reshape(1, -1)

    model_name = Path(model_path).name.lower()
    if "svm" in model_name:
        s_path = Path("experiments/checkpoints/scaler_fruit.pkl")
        if not s_path.exists():
            s_path = Path(scaler_path)
        scaler = load_pickle(s_path)
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

    # Calcular el diámetro
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        diameter = float(max(w, h))
    else:
        h_img, w_img = img_cropped.shape[:2]
        diameter = float(max(w_img, h_img))

    model = load_pickle(model_path)
    input_features = features.reshape(1, -1)

    model_name = Path(model_path).name.lower()
    if "svm" in model_name:
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
    
    size_result = estimate_size(features)

    return{
        "model_type": "traditional_ml",
        "model_path": str(model_path),
        "quality": QUALITY_LABELS.get(predicted_id, str(predicted_id)),
        "quality_id": predicted_id,
        "confidence": confidence,
        "probabilities": probabilities,
        "diameter": diameter,
        **size_result,
        **fruit_result,
    }

def predict_with_CNN(
        image_path,
        model_path = "experiments/checkpoints/best_model.pth",
        fruit_model_path = "experiments/checkpoints/best_fruit_model.pth",
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
        size_result = estimate_size(features)

        # Calcular el diámetro
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            diameter = float(max(w, h))
        else:
            h_img, w_img = img_cropped.shape[:2]
            diameter = float(max(w_img, h_img))

        fruit_model_path = Path(fruit_model_path)
        if fruit_model_path.exists():
            fruit_model, _ = load_cnn(model_pah=fruit_model_path, num_classes=6, device=device)
            
            with torch.no_grad():
                fruit_logits = fruit_model(tensor)
                fruit_probs = torch.softmax(fruit_logits, dim=1)[0]
                fruit_predicted_id = int(torch.argmax(fruit_probs).item())
                
            fruit_confidence = float(fruit_probs[fruit_predicted_id].cpu().item())
            fruit_probabilities = {
                FRUIT_LABELS.get(idx, str(idx)): float(fruit_probs[idx].cpu().item())
                for idx in range(len(FRUIT_LABELS))
            }
            
            fruit_result = {
                "fruit": FRUIT_LABELS.get(fruit_predicted_id, str(fruit_predicted_id)),
                "fruit_id": fruit_predicted_id,
                "fruit_confidence": fruit_confidence,
                "fruit_probabilities": fruit_probabilities,
            }
        else:
            fruit_result = predict_fruit_type_from_features(features)

        return {
            "model_type": "cnn",
            "model_path": str(model_path),
            "quality": QUALITY_LABELS.get(predicted_id, str(predicted_id)),
            "quality_id": predicted_id,
            "confidence": float(probabilities_tensor[predicted_id].cpu().item()),
            "probabilities": probabilities,
            "diameter": diameter,
            **size_result,
            **fruit_result,
        }


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
    result = predict_image("test_image.jpj", model_type="random_forest")
    print(result)