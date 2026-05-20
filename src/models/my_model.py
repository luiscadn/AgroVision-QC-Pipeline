from src.utils.helpers import setup_logger

logger = setup_logger("AgroVision-Model")

def create_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2):
    """
    Define y compila la estructura de la Red Neuronal Convolucional (CNN).
    
    Args:
        input_shape (tuple): Dimensiones de la imagen de entrada.
        num_classes (int): Número de clases de salida.
        
    Returns:
        tf.keras.Model: Modelo compilado (si tensorflow está disponible).
    """
    logger.info("Definiendo arquitectura CNN...")
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1 if num_classes == 2 else num_classes, 
                         activation='sigmoid' if num_classes == 2 else 'softmax')
        ])
        
        loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        logger.info("Modelo CNN compilado correctamente.")
        return model
    except ImportError:
        logger.warning("TensorFlow no está instalado. Devolviendo estructura simulada en su lugar.")
        return {"input_shape": input_shape, "num_classes": num_classes, "status": "Simulado (Instale tensorflow)"}
