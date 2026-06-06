from src.models.predict import estimate_size
import numpy as np


def test_estimate_size_returns_valid_label():
    features = np.array([1000, 120, 1.0, 30, 40, 50, 5, 6, 7], dtype=np.float32)
    result = estimate_size(features, reference_csv="archivo_que_no_existe.csv")
    assert result["size"] in {"pequeño", "mediano", "grande"}
    assert result["area_px"] == 1000.0
