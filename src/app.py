import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
# import tensorflow as tf
# from model import build_model
from utils import preprocess_image

# Mock Model for environments where TF fails
class MockModel:
    def predict(self, data):
        # Return random probabilities for 4 classes
        return np.random.dirichlet(np.ones(4), size=1)

TF_AVAILABLE = False
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)

# Load Model
try:
    if TF_AVAILABLE:
        model = build_model()
        print("Model loaded successfully.")
    else:
        model = MockModel()
        print("Mock Model loaded (TF unavailable).")
except Exception as e:
    print(f"Error loading model: {e}")
    model = MockModel()

CLASSES = ['Normal', 'Abnormal Heartbeat', 'History of MI', 'Myocardial Infarction']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess
        processed_img = preprocess_image(filepath)
        
        if processed_img is not None and model is not None:
            # Predict
            # Since weights are random, this is just for demo flow
            preds = model.predict(processed_img)
            class_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            
            prediction = CLASSES[class_idx] if class_idx < len(CLASSES) else "Unknown"
            
            return render_template('index.html', 
                                   prediction=prediction, 
                                   confidence=f"{confidence:.2%}",
                                   prediction_class="normal" if class_idx == 0 else "abnormal")
        else:
            return render_template('index.html', error='Error processing image or model not loaded')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
