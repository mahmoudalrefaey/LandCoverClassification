import streamlit as st
import sys
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import io
from config import DATA_CONFIG

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model_handler import ModelHandler
from data_processor import DataProcessor

class WebApp:
    def __init__(self):
        self.model_handler = ModelHandler()
        self.data_processor = DataProcessor()
        
        # Load model and class indices
        if not self.model_handler.load_model():
            st.error("Failed to load model. Please check if ResNet50_eurosat.h5 exists.")
        if not self.model_handler.load_class_indices():
            st.error("Failed to load class indices. Please check if class_indices.npy exists.")
        
    def main_page(self):
        st.title("Land Cover Classification System")
        st.write("Upload a satellite image to classify its land cover type.")
        
        # Display class information
        with st.expander("Available Land Cover Classes"):
            st.write("0 - AnnualCrop")
            st.write("1 - Forest")
            st.write("2 - HerbaceousVegetation")
            st.write("3 - Highway")
            st.write("4 - Industrial")
            st.write("5 - Pasture")
            st.write("6 - PermanentCrop")
            st.write("7 - Residential")
            st.write("8 - River")
            st.write("9 - SeaLake")
        
        # Image upload section
        uploaded_image = self.upload()
        
        if uploaded_image is not None:
            # Create two columns for image preview and controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image Preview")
                st.image(uploaded_image, use_column_width=True)
            
            with col2:
                st.subheader("Classification")
                if st.button('Run Classification'):
                    self.run_classification(uploaded_image)

    def upload(self):
        """Handle image upload"""
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=DATA_CONFIG['allowed_formats']
        )
        
        if uploaded_file is not None:
            return uploaded_file.read()
        return None

    def run_classification(self, image_data):
        """Run classification on uploaded image"""
        try:
            with st.spinner('Running classification...'):
                # Get prediction
                prediction = self.model_handler.predict(image_data)
                
                # Display results
                st.success(f"Classification Result: {prediction['class_name']}")
                st.write(f"Confidence: {prediction['confidence']:.2%}")
                
                # Create confidence bar chart
                self.plot_confidence(prediction['all_predictions'])
                
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")

    def plot_confidence(self, predictions):
        """Plot confidence scores for all classes"""
        # Get class names from indices
        class_names = [self.model_handler.class_indices.get(str(i), f"Class_{i}") 
                      for i in range(len(predictions))]
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Class': class_names,
            'Confidence': predictions
        })
        
        # Sort by confidence
        df = df.sort_values('Confidence', ascending=False)
        
        # Create bar chart
        fig = px.bar(df, 
                    x='Class', 
                    y='Confidence',
                    title='Classification Confidence Scores',
                    labels={'Confidence': 'Confidence Score'},
                    color='Confidence',
                    color_continuous_scale='Viridis')
        
        # Update layout
        fig.update_layout(
            xaxis_title="Land Cover Class",
            yaxis_title="Confidence Score",
            yaxis_tickformat='.1%',
            showlegend=False
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)

# Initialize and run the app
if __name__ == "__main__":
    app = WebApp()
    app.main_page()