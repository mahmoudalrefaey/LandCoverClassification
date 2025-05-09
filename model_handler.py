import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from PIL import Image
import io
from config import CLASS_NAMES, MODEL_CONFIG

class ModelHandler:
    def __init__(self):
        self.model = None
        self.class_indices = CLASS_NAMES
        self.input_shape = MODEL_CONFIG['input_shape']
        
    def load_model(self, model_path=MODEL_CONFIG['model_path']):
        """Load the pre-trained ResNet50 model"""
        try:
            self.model = load_model(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
            
    def load_class_indices(self, indices_path=MODEL_CONFIG['indices_path']):
        """Load class indices mapping"""
        try:
            # Load indices from file if it exists
            if os.path.exists(indices_path):
                loaded_indices = np.load(indices_path, allow_pickle=True).item()
                # Update class indices with loaded values
                self.class_indices.update(loaded_indices)
            return True
        except Exception as e:
            print(f"Error loading class indices: {str(e)}")
            return False
            
    def preprocess_image(self, image):
        """Preprocess image for model input"""
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
        
    def predict(self, image):
        """Make prediction on input image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        predictions = self.model.predict(processed_image)
        
        # Get top prediction
        top_pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][top_pred_idx]
        
        # Get class name from indices
        class_name = self.class_indices.get(str(top_pred_idx), f"Class_{top_pred_idx}")
        
        return {
            'class_name': class_name,
            'confidence': float(confidence),
            'all_predictions': predictions[0].tolist()
        }

    @staticmethod
    def compile_model(input_shape, n_classes, optimizer, fine_tune=None):
        """Compile a new ResNet50 model (for training)"""
        conv_base = ResNet50(include_top=False,
                            weights='imagenet',
                            input_shape=input_shape)

        top_model = conv_base.output
        top_model = Flatten()(top_model)
        top_model = Dense(2048, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes, activation='softmax')(top_model)

        model = Model(inputs=conv_base.input, outputs=output_layer)

        if isinstance(fine_tune, int):
            for layer in conv_base.layers[fine_tune:]:
                layer.trainable = True
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])

        return model