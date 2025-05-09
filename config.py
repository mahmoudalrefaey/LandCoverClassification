"""
Configuration file for the Land Cover Classification System
"""

# EuroSAT class names
CLASS_NAMES = {
    '0': 'AnnualCrop',
    '1': 'Forest',
    '2': 'HerbaceousVegetation',
    '3': 'Highway',
    '4': 'Industrial',
    '5': 'Pasture',
    '6': 'PermanentCrop',
    '7': 'Residential',
    '8': 'River',
    '9': 'SeaLake'
}

# Model configuration
MODEL_CONFIG = {
    'input_shape': (64, 64, 3),
    'model_path': 'models/ResNet50_eurosat.h5',
    'best_model_path': 'models/model.weights.best.keras',
    'indices_path': 'models/class_indices.npy'
}

# Data processing configuration
DATA_CONFIG = {
    'allowed_formats': ['png', 'jpg', 'jpeg', 'tiff'],
    'max_image_size': (1024, 1024)  # Maximum dimensions for uploaded images
} 