import unittest
import os
import sys

# Asegurar que el directorio raíz está en el PYTHONPATH para las pruebas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import setup_logger, get_project_root
from src.models.my_model import create_cnn_model

class TestAgroVisionPipeline(unittest.TestCase):
    
    def test_logger_setup(self):
        logger = setup_logger("TestLogger")
        self.assertIsNotNone(logger)
        
    def test_project_root(self):
        root = get_project_root()
        self.assertTrue(os.path.isdir(root))
        
    def test_model_creation(self):
        model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
