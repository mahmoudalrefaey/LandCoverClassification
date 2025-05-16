import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import cv2
import pandas as pd
from PIL import Image
import io
import streamlit as st

class Visualizer:
    def __init__(self):
        pass
        
    def plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt
        
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        return plt

    def plot_confidence_bar(self, class_names, predictions):
        df = pd.DataFrame({
            'Class': class_names,
            'Confidence': predictions
        }).sort_values('Confidence', ascending=False)
        fig = px.bar(df, 
                    x='Class', 
                    y='Confidence',
                    title='Classification Confidence Scores',
                    labels={'Confidence': 'Confidence Score'},
                    color='Confidence',
                    color_continuous_scale='Viridis')
        fig.update_layout(
            xaxis_title="Land Cover Class",
            yaxis_title="Confidence Score",
            yaxis_tickformat='.1%',
            showlegend=False
        )
        return fig

    def plot_rgb_histograms(self, img_array):
        colors = ['Red', 'Green', 'Blue']
        figs = []
        for i, color in enumerate(colors):
            fig, ax = plt.subplots()
            histogram = np.histogram(img_array[:,:,i], bins=256, range=(0,256))[0]
            ax.plot(histogram, color=color.lower(), alpha=0.8)
            ax.set_title(f"{color} Channel")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            figs.append(fig)
        return figs

    def image_statistics(self, img_array):
        stats = {
            "Mean Brightness": float(np.mean(img_array)),
            "Standard Deviation": float(np.std(img_array)),
            "Min Value": int(np.min(img_array)),
            "Max Value": int(np.max(img_array)),
            "Image Size": f"{img_array.shape[1]}x{img_array.shape[0]}",
            "Channels": img_array.shape[2]
        }
        return stats

    def edge_detection(self, img_array):
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def intensity_map(self, img_array):
        gray = np.mean(img_array, axis=2)
        fig = px.imshow(gray, 
                        title="Intensity Map",
                        color_continuous_scale='viridis')
        return fig
