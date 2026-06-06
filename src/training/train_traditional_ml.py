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
    
    X = df.drop(columns=['label', 'fruit_label'], errors='ignore').values
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
    svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=3, n_jobs=-1, verbose=1)
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
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1, verbose=1)
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
    
    # ------------------------------------------------------------------
    # ENTRENAMIENTO DE CLASIFICACIÓN DE TIPO DE FRUTA
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("   Entrenamiento de Clasificadores de Tipo de Fruta (Tabular)")
    logger.info("=" * 60)

    # 1. Extraer objetivo de fruta
    y_fruit = df['fruit_label'].values
    
    # Nombres de clase de fruta
    fruit_class_names = ['manzana', 'banano', 'guayaba', 'limon', 'naranja', 'granada']
    
    # Partición Estratificada para fruta
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X, y_fruit, test_size=0.3, random_state=42, stratify=y_fruit
    )
    
    # Escalar para SVM de frutas
    scaler_fruit = StandardScaler()
    X_train_f_scaled = scaler_fruit.fit_transform(X_train_f)
    X_test_f_scaled = scaler_fruit.transform(X_test_f)
    
    # Guardar escalador de fruta
    scaler_fruit_path = os.path.join(checkpoints_dir, "scaler_fruit.pkl")
    with open(scaler_fruit_path, 'wb') as f:
        pickle.dump(scaler_fruit, f)
    logger.info(f"Escalador de fruta guardado en: {scaler_fruit_path}")
    
    # Entrenar SVM de frutas
    logger.info("\n--- Entrenando SVM para Tipo de Fruta ---")
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_grid_f = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=3, n_jobs=-1, verbose=1)
    svm_grid_f.fit(X_train_f_scaled, y_train_f)
    best_svm_f = svm_grid_f.best_estimator_
    acc_svm_f = accuracy_score(y_test_f, best_svm_f.predict(X_test_f_scaled))
    logger.info(f"Accuracy SVM Fruta en Test: {acc_svm_f:.4f}")
    
    # Entrenar RF de frutas
    logger.info("\n--- Entrenando Random Forest para Tipo de Fruta ---")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf_grid_f = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1, verbose=1)
    rf_grid_f.fit(X_train_f, y_train_f)
    best_rf_f = rf_grid_f.best_estimator_
    acc_rf_f = accuracy_score(y_test_f, best_rf_f.predict(X_test_f))
    logger.info(f"Accuracy Random Forest Fruta en Test: {acc_rf_f:.4f}")
    
    # Guardar el MEJOR modelo para tipo de fruta
    if acc_rf_f >= acc_svm_f:
        best_fruit_model = best_rf_f
        best_model_name = "Random Forest"
        logger.info(f"El mejor modelo de fruta fue Random Forest (Acc: {acc_rf_f:.4f})")
    else:
        best_fruit_model = best_svm_f
        best_model_name = "SVM"
        logger.info(f"El mejor modelo de fruta fue SVM (Acc: {acc_svm_f:.4f})")
        
    fruit_model_path = os.path.join(checkpoints_dir, "fruit_type_model.pkl")
    with open(fruit_model_path, 'wb') as f:
        pickle.dump(best_fruit_model, f)
    logger.info(f"Modelo de fruta guardado en: {fruit_model_path}")
    
    # Guardar matrices de confusión y métricas en JSON
    _save_confusion_matrix(y_test_f, best_svm_f.predict(X_test_f_scaled), fruit_class_names, os.path.join(results_dir, "confusion_matrix_fruit_svm.png"), "SVM - Tipo de Fruta")
    _save_confusion_matrix(y_test_f, best_rf_f.predict(X_test_f), fruit_class_names, os.path.join(results_dir, "confusion_matrix_fruit_rf.png"), "Random Forest - Tipo de Fruta")
    
    # Guardar métricas en JSON
    fruit_metrics = {
        "svm_fruit": {
            "best_params": svm_grid_f.best_params_,
            "accuracy": float(acc_svm_f)
        },
        "random_forest_fruit": {
            "best_params": rf_grid_f.best_params_,
            "accuracy": float(acc_rf_f)
        },
        "best_overall": best_model_name
    }
    
    fruit_metrics_path = os.path.join(results_dir, "traditional_ml_fruit_metrics.json")
    with open(fruit_metrics_path, "w", encoding="utf-8") as f:
        json.dump(fruit_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas de ML fruta guardadas en: {fruit_metrics_path}")
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
