from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pickle
from PIL import Image
from torchvision import transforms
from src.data.preprocess import load_and_segment_fruit, get_traditional_features
from src.models.cnn_model import FruitQualityCNN
from src.models.fruit_classifier import load_fruit_classifier


# Clases de calidad
# ⚠️ IMPORTANTE: ImageFolder ordena las carpetas alfabéticamente.
# Orden real en training: buena=0, mala=1, media=2
QUALITY_LABELS = {
    0: "buena",
    1: "mala",
    2: "media",
}

# Clases de fruta
# ⚠️ IMPORTANTE: ImageFolder ordena alfabéticamente las carpetas.
# Orden real: banano=0, granada=1, guayaba=2, limon=3, manzana=4, naranja=5
FRUIT_LABELS = {
    0: "banano",
    1: "granada",
    2: "guayaba",
    3: "limon",
    4: "manzana",
    5: "naranja",
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
) -> Tuple[FruitQualityCNN, torch.device]:
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
    """Predice qué fruta usando características tradicionales.
    Si el modelo pkl no existe, devuelve 'desconocida' en vez de lanzar error.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        # Fallback cuando no hay modelo de tipo de fruta entrenado
        return {
            "fruit": "desconocida",
            "fruit_id": -1,
            "fruit_confidence": None,
            "fruit_probabilities": None,
        }

    model = load_pickle(model_path)
    input_features = features.reshape(1, -1)

    model_name = model_path.name.lower()
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

# Aquí hacemos la predicción usando un modelo tradicional
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

    return {
        "model_type": "traditional_ml",
        "model_path": str(model_path),
        "quality": QUALITY_LABELS.get(predicted_id, str(predicted_id)),
        "quality_id": predicted_id,
        "confidence": confidence,
        "probabilities": probabilities,
        **size_result,
        **fruit_result,
    }

def predict_with_CNN(
        image_path,
        model_path = "experiments/checkpoints/best_model.pth",
        fruit_model_path = "experiments/checkpoints/best_fruit_model.pth",
        device: Optional[str] = None,
) -> Dict[str, object]:
    """Predice calidad (FruitQualityCNN 128x128) y tipo de fruta (MobileNetV2 224x224)."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")

    selected_device = torch.device("cpu")

    # ── 1. CALIDAD con FruitQualityCNN (128x128) ─────────────────────────────
    model, _ = load_cnn(model_pah=model_path, device=device)

    quality_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_pil = Image.open(image_path).convert("RGB")
    quality_tensor = quality_transform(image_pil).unsqueeze(0).to(selected_device)

    with torch.no_grad():
        quality_logits = model(quality_tensor)
        quality_probs  = torch.softmax(quality_logits, dim=1)[0]
        predicted_id   = int(torch.argmax(quality_probs).item())

    probabilities = {
        QUALITY_LABELS[i]: float(quality_probs[i].cpu().item())
        for i in range(len(QUALITY_LABELS))
    }

    # ── 2. TAMAÑO usando características clásicas ─────────────────────────────
    img_cropped, contour = load_and_segment_fruit(str(image_path))
    features = get_traditional_features(img_cropped, contour)
    size_result = estimate_size(features)

    # ── 3. TIPO DE FRUTA con MobileNetV2 (224x224) ───────────────────────────
    fruit_model_path = Path(fruit_model_path)
    if fruit_model_path.exists():
        fruit_model = load_fruit_classifier(
            checkpoint_path=str(fruit_model_path),
            num_classes=6,
            device=selected_device,
        )

        fruit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        fruit_tensor = fruit_transform(image_pil).unsqueeze(0).to(selected_device)

        with torch.no_grad():
            fruit_logits = fruit_model(fruit_tensor)
            fruit_probs  = torch.softmax(fruit_logits, dim=1)[0]
            fruit_id     = int(torch.argmax(fruit_probs).item())

        fruit_result = {
            "fruit": FRUIT_LABELS.get(fruit_id, str(fruit_id)),
            "fruit_id": fruit_id,
            "fruit_confidence": float(fruit_probs[fruit_id].cpu().item()),
            "fruit_probabilities": {
                FRUIT_LABELS.get(i, str(i)): float(fruit_probs[i].cpu().item())
                for i in range(len(FRUIT_LABELS))
            },
        }
    else:
        fruit_result = predict_fruit_type_from_features(features)

    return {
        "model_type": "cnn",
        "model_path": str(model_path),
        "quality": QUALITY_LABELS.get(predicted_id, str(predicted_id)),
        "quality_id": predicted_id,
        "confidence": float(quality_probs[predicted_id].cpu().item()),
        "probabilities": probabilities,
        **size_result,
        **fruit_result,
    }


def predict_image(
    image_path: str | Path,
    model_type: str = "random_forest",
) -> Dict[str, object]:
    # Aquí el tipo de modelo puede ser cualquiera: randomF, svm o cnn
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
    result = predict_image("test_image.jpg", model_type="cnn")
    print(result)