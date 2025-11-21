# CardioVision: AI-Powered Cardiovascular Disease Detector

CardioVision is a web application designed to detect cardiovascular abnormalities from ECG images using deep learning techniques (CNN-LSTM).

## Features

-   **ECG Analysis**: Upload ECG images to detect potential cardiovascular diseases.
-   **Deep Learning Model**: Utilizes a CNN-LSTM architecture for feature extraction and sequence modeling.
-   **User-Friendly Interface**: Simple and clean web interface built with Flask.
-   **Real-time Prediction**: Instant analysis results.

## Project Structure

```
.
├── requirements.txt    # Project dependencies
├── src/
│   ├── app.py          # Flask application entry point
│   ├── model.py        # CNN-LSTM model definition (TensorFlow/Keras)
│   ├── utils.py        # Image preprocessing utilities
│   ├── static/         # Static assets (CSS, uploads)
│   └── templates/      # HTML templates
└── README.md           # Project documentation
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ashmitha-14/Direction-of-cardiovascular-diseases-by-ECG-images-using-machine-learning-and-deep-learning.git
    cd Direction-of-cardiovascular-diseases-by-ECG-images-using-machine-learning-and-deep-learning
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    python src/app.py
    ```

2.  **Access the Web UI**:
    Open your web browser and navigate to `http://127.0.0.1:5000`.

3.  **Analyze an ECG**:
    -   Click "Choose an ECG Image" to upload a file.
    -   Click "Analyze ECG".
    -   View the prediction results.

## Note on Model

This repository contains the model architecture definition in `src/model.py`. By default, the application may run in a "Mock Mode" if TensorFlow is not properly configured or if trained weights are missing, providing random predictions for demonstration purposes. To use the actual model, ensure TensorFlow is installed and load your trained weights in `src/app.py`.

## Technologies

-   **Backend**: Flask (Python)
-   **ML/DL**: TensorFlow, Keras, OpenCV, NumPy, Pandas
-   **Frontend**: HTML5, CSS3
