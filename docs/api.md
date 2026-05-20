# Documentación de la API

Este documento describe la especificación de la interfaz programática y los endpoints en caso de exponer el pipeline de control de calidad como un servicio web.

## Endpoints Principales

### 1. Clasificar Calidad de Fruta
Realiza la predicción de calidad (Aceptado/Rechazado) a partir de una imagen enviada.

* **URL**: `/api/v1/predict`
* **Método**: `POST`
* **Content-Type**: `multipart/form-data`
* **Parámetros de Body**:
  * `image`: Archivo de imagen (formatos válidos: `.jpg`, `.jpeg`, `.png`).
* **Respuesta Exitosa (200 OK)**:
  ```json
  {
    "status": "success",
    "filename": "apple_sample_12.jpg",
    "prediction": "Rechazado",
    "confidence": 0.945,
    "defect_detected": "Mancha superficial de hongos",
    "processing_time_ms": 142.5
  }
  ```

### 2. Estado del Modelo
Obtiene información sobre la versión del modelo actualmente desplegado y las métricas del pipeline.

* **URL**: `/api/v1/model/status`
* **Método**: `GET`
* **Respuesta Exitosa (200 OK)**:
  ```json
  {
    "model_name": "AgroVision-CNN-FruitQC",
    "version": "1.2.0",
    "last_trained": "2026-05-19T10:30:00Z",
    "framework": "TensorFlow 2.14",
    "accuracy": 0.967
  }
  ```
