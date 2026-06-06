import os
import sys
import pickle

# Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-TrainML")

def train_traditional_ml(
    features_csv: str = "experiments/results/features_traditional_ml.csv",
    checkpoints_dir: str = "experiments/checkpoints",
    results_dir: str = "experiments/results"
):
    """
    Entrena y optimiza dos modelos de Machine Learning tradicional (SVM y Random Forest)
    a partir de las características extraídas (Fase 4 de CRISP-DM).
    """
    logger.info("=" * 60)
    logger.info("   Entrenamiento de Modelos de Machine Learning Tradicional")
    logger.info("=" * 60)

    if not os.path.exists(features_csv):
        logger.error(f"No se encontró el archivo de características en '{features_csv}'. "
                     "Por favor, corre la Fase 1 del pipeline primero.")
        return

    # 1. Cargar características
    logger.info(f"Cargando características desde {features_csv}...")
    df = pd.read_csv(features_csv)
    
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # Mapeo de etiquetas numéricas a nombres legibles
    class_names = ['buena', 'media', 'mala']

    # 2. Partición Estratificada (70% Train, 30% Test para validación del modelo tradicional)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logger.info(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    logger.info(f"Datos de prueba: {X_test.shape[0]} muestras")

    # 3. Escalar características (Crucial para SVM y algoritmos basados en distancias)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar el escalador para uso futuro en producción/despliegue
    scaler_path = os.path.join(checkpoints_dir, "scaler_ml.pkl")
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Escalador guardado en: {scaler_path}")

    # ------------------------------------------------------------------
    # MODELO 1: Máquina de Soporte Vectorial (SVM) con Búsqueda en Rejilla
    # ------------------------------------------------------------------
    logger.info("\n--- Entrenando SVM con GridSearchCV ---")
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(SVC(probability=True, random_state=42, class_weight="balanced"), svm_param_grid, cv=3, n_jobs=-1, verbose=1)
    svm_grid.fit(X_train_scaled, y_train)
    
    best_svm = svm_grid.best_estimator_
    logger.info(f"Mejores hiperparámetros SVM: {svm_grid.best_params_}")

    # Evaluación SVM
    y_pred_svm = best_svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    logger.info(f"Accuracy SVM en Test: {acc_svm:.4f}")
    logger.info("\nReporte de Clasificación SVM:\n" + classification_report(y_test, y_pred_svm, target_names=class_names))

    # Guardar modelo SVM
    svm_path = os.path.join(checkpoints_dir, "svm_model.pkl")
    with open(svm_path, 'wb') as f:
        pickle.dump(best_svm, f)
    logger.info(f"Modelo SVM guardado en: {svm_path}")

    # ------------------------------------------------------------------
    # MODELO 2: Random Forest con Búsqueda en Rejilla
    # ------------------------------------------------------------------
    logger.info("\n--- Entrenando Random Forest con GridSearchCV ---")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"), rf_param_grid, cv=3, n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)  # Árboles no requieren escalado
    
    best_rf = rf_grid.best_estimator_
    logger.info(f"Mejores hiperparámetros Random Forest: {rf_grid.best_params_}")

    # Evaluación Random Forest
    y_pred_rf = best_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    logger.info(f"Accuracy Random Forest en Test: {acc_rf:.4f}")
    logger.info("\nReporte de Clasificación Random Forest:\n" + classification_report(y_test, y_pred_rf, target_names=class_names))

    # Guardar modelo Random Forest
    rf_path = os.path.join(checkpoints_dir, "random_forest_model.pkl")
    with open(rf_path, 'wb') as f:
        pickle.dump(best_rf, f)
    logger.info(f"Modelo Random Forest guardado en: {rf_path}")

    # ------------------------------------------------------------------
    # Guardar resultados en disco
    # ------------------------------------------------------------------
    # Guardar matrices de confusión para reporte final
    _save_confusion_matrix(y_test, y_pred_svm, class_names, os.path.join(results_dir, "confusion_matrix_svm.png"), "SVM")
    _save_confusion_matrix(y_test, y_pred_rf, class_names, os.path.join(results_dir, "confusion_matrix_rf.png"), "Random Forest")

    # Guardar métricas en JSON
    metrics = {
        "svm": {
            "best_params": svm_grid.best_params_,
            "accuracy": float(acc_svm)
        },
        "random_forest": {
            "best_params": rf_grid.best_params_,
            "accuracy": float(acc_rf)
        }
    }
    
    metrics_path = os.path.join(results_dir, "traditional_ml_metrics.json")
    import json
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas de ML tradicional guardadas en: {metrics_path}")
    logger.info("=" * 60)

def _save_confusion_matrix(y_true, y_pred, class_names, output_path, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"Matriz de Confusión - {model_name}",
        ylabel="Etiqueta Real",
        xlabel="Etiqueta Predicha"
    )

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i,
                f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    train_traditional_ml()
