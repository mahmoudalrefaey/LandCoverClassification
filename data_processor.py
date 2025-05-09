import os
import hashlib
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import io

class DataProcessor:
    def __init__(self):
        self.input_shape = (64, 64, 3)  # Default input shape for EuroSAT
        
    def preprocess_for_inference(self, image):
        """Preprocess a single image for model inference"""
        if isinstance(image, bytes):
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image))
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Convert PIL Image to numpy array
            image = np.array(image)
            
            # Convert to float32 and normalize
            image = image.astype(np.float32) / 255.0
            
            # Resize image
            image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]))
            
            # Add batch dimension
            image = tf.expand_dims(image, 0)
            
            return image
        else:
            raise ValueError("Input must be bytes (image file content)")

    @staticmethod
    def check_image_size(image_path):
        """Check dimensions of an image file"""
        with Image.open(image_path) as img:
            return img.size

    @staticmethod
    def check_image_dimensions(dataset_path):
        """Check dimensions of all images in dataset"""
        all_dimensions = set()
        for folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    width, height = DataProcessor.check_image_size(image_path)
                    all_dimensions.add((width, height))
        return all_dimensions

    @staticmethod
    def get_data_generators():
        """Get data generators for training and validation"""
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=60,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        test_gen = ImageDataGenerator(rescale=1./255)
        
        return train_gen, test_gen

    @staticmethod
    def get_image_hash(image_path):
        """Calculate MD5 hash of an image file"""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def check_duplicates(dataset_path):
        """Find duplicate images in dataset"""
        seen_hashes = set()
        duplicates = []
        for folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    img_hash = DataProcessor.get_image_hash(image_path)
                    if img_hash in seen_hashes:
                        duplicates.append(image_path)
                    else:
                        seen_hashes.add(img_hash)
        return duplicates