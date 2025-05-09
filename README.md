# LandCoverClassification

A web-based land cover classification system using a pretrained ResNet50 model and the EuroSAT dataset. The project features a modern Streamlit interface for easy image upload, prediction, and visualization of land type probabilities.

## Features

- **Deep Learning Model**: Utilizes a pretrained ResNet50 model fine-tuned on EuroSAT satellite imagery.
- **User-Friendly Web App**: Built with Streamlit for interactive image upload and real-time predictions.
- **Class Visualization**: Displays prediction confidence for each land cover class using intuitive bar charts.
- **Configurable & Modular**: Clean separation of model handling, data processing, and visualization logic.

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mahmoudalrefaey/LandCoverClassification.git
   cd LandCoverClassification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model & Class Indices:**
   - Place `ResNet50_eurosat.h5` and `class_indices.npy` in the project root directory.

### Running the App

```bash
streamlit run app.py
```

Open your browser and go to the provided local URL (usually http://localhost:8501).

## Usage

1. **Upload a satellite image** (formats: PNG, JPG, JPEG, TIFF).
2. **Preview the image** and click "Run Classification".
3. **View the predicted land cover class** and the confidence scores for all classes.

## Land Cover Classes

| Index | Class Name             |
|-------|------------------------|
| 0     | AnnualCrop             |
| 1     | Forest                 |
| 2     | HerbaceousVegetation   |
| 3     | Highway                |
| 4     | Industrial             |
| 5     | Pasture                |
| 6     | PermanentCrop          |
| 7     | Residential            |
| 8     | River                  |
| 9     | SeaLake                |

## Project Structure

```
.
├── app.py                # Streamlit web app
├── model_handler.py      # Model loading and prediction logic
├── data_processor.py     # Data preprocessing utilities
├── config.py             # Configuration (class names, paths)
├── requirements.txt      # Python dependencies
├── ResNet50_eurosat.h5   # Pretrained model weights (not included)
├── class_indices.npy     # Class index mapping (not included)
└── README.md             # This file
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Repository:** [@mahmoudalrefaey/LandCoverClassification](https://github.com/mahmoudalrefaey/LandCoverClassification) 