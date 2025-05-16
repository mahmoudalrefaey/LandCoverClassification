import streamlit as st
import sys
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io
import cv2
from config import DATA_CONFIG, CLASS_NAMES

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model_handler import ModelHandler
from data_processor import DataProcessor
from visualizer import Visualizer

class WebApp:
    def __init__(self):
        self.model_handler = ModelHandler()
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        
        # Load model and class indices
        if not self.model_handler.load_model():
            st.error("Failed to load model. Please check if models/ResNet50_eurosat.h5 exists.")
        if not self.model_handler.load_class_indices():
            st.error("Failed to load class indices. Please check if models/class_indices.npy exists.")
    
    def main_page(self):
        st.title("Project DEPI - Land Cover Classification")
        st.caption("Welcome to the Land Cover Classification System")
        
        # Display welcome image
        st.image("assets/satellite.jpg", use_column_width=True)
        
        # Image upload section
        uploaded_image = self.upload()
        
        if uploaded_image is not None:
            # Store the uploaded image in session state
            st.session_state.uploaded_image = uploaded_image
            
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
        return uploaded_file.read() if uploaded_file is not None else None

    def run_classification(self, image_data):
        """Run classification on uploaded image"""
        try:
            with st.spinner('Running classification...'):
                prediction = self.model_handler.predict(image_data)
                st.success(f"Classification Result: {prediction['class_name']}")
                st.write(f"Confidence: {prediction['confidence']:.2%}")
                # Use visualizer for confidence bar
                class_names = [self.model_handler.class_indices.get(str(i), f"Class_{i}") 
                              for i in range(len(prediction['all_predictions']))]
                fig = self.visualizer.plot_confidence_bar(class_names, prediction['all_predictions'])
                st.plotly_chart(fig, use_column_width=True)
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")

    def charts_page(self):
        """Display various charts and visualizations"""
        st.title("Charts and Visualizations")
        
        tab1, tab2 = st.tabs(["Model evaluation", "Image Analysis"])
        
        with tab1:
            st.subheader("Model Performance")
            st.title("Model Training Results")
            st.image("assets/model_performance.jpg", caption="Training Progress Over Time", use_column_width=True)
            with st.expander("Accuracy Analysis"):
                st.markdown("""
                - **Training Accuracy**: Shows how well the model learns from training data
                - **Validation Accuracy**: Indicates real-world performance on unseen data
                - **Ideal Scenario**: Both metrics should increase and stabilize at similar values
                """)
            with st.expander("Loss Analysis"):
                st.markdown("""
                - **Training Loss**: Measures error reduction during training
                - **Validation Loss**: Tracks generalization error
                - **Healthy Pattern**: Both should decrease steadily without significant divergence
                """)
            st.header("What This Means")
            st.write("""
            The model is learning properly without overfitting.
            Both accuracy and loss show good progress.
            You could stop training earlier when it stops improving.
            """)
        with tab2:
            st.subheader("Image Analysis")
            if 'uploaded_image' in st.session_state:
                uploaded_image = st.session_state.uploaded_image
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                try:
                    img = Image.open(io.BytesIO(uploaded_image))
                    img_array = np.array(img)
                    # Use visualizer for RGB histograms
                    st.subheader("RGB Color Histograms")
                    figs = self.visualizer.plot_rgb_histograms(img_array)
                    col1, col2, col3 = st.columns(3)
                    for i, col in enumerate([col1, col2, col3]):
                        with col:
                            st.pyplot(figs[i])
                    # Visualization selector
                    analysis_type = st.selectbox(
                        "Select Analysis Type",
                        ["Image Statistics", "Edge Detection", "Intensity Map"]
                    )
                    if analysis_type == "Image Statistics":
                        stats = self.visualizer.image_statistics(img_array)
                        for key, value in stats.items():
                            st.write(f"**{key}:** {value}")
                    elif analysis_type == "Edge Detection":
                        edges = self.visualizer.edge_detection(img_array)
                        st.image(edges, caption="Edge Detection", use_column_width=True)
                    elif analysis_type == "Intensity Map":
                        fig = self.visualizer.intensity_map(img_array)
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            else:
                st.warning("Please upload an image in the main page first.")

    def classes_page(self):
        """Display detailed information about each class"""
        st.title("Land Cover Classes")
        
        # List of (class_name, description)
        class_info = [
            ("AnnualCrop", "Agricultural areas where crops are planted and harvested within a single year."),
            ("Forest", "Areas dominated by trees, forming a continuous canopy."),
            ("HerbaceousVegetation", "Areas covered by non-woody plants and grasses."),
            ("Highway", "Major roads and transportation infrastructure."),
            ("Industrial", "Areas containing factories, warehouses, and industrial facilities."),
            ("Pasture", "Land used for grazing livestock."),
            ("PermanentCrop", "Agricultural areas with long-term crops like orchards and vineyards."),
            ("Residential", "Areas containing houses and residential buildings."),
            ("River", "Natural watercourses and their immediate surroundings."),
            ("SeaLake", "Large bodies of water including seas and lakes.")
        ]
        
        cols = st.columns(2)
        for idx, (name, desc) in enumerate(class_info):
            with cols[idx % 2]:
                with st.expander(f"{name}"):
                    st.write(desc)

# Initialize and run the app
if __name__ == "__main__":
    app = WebApp()
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Charts", "Classes"])
    
    if page == "Home":
        app.main_page()
    elif page == "Charts":
        app.charts_page()
    elif page == "Classes":
        app.classes_page()