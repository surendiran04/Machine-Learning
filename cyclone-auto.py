import os
import numpy as np
import pandas as pd
from PIL import Image  # Import Pillow's Image module
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request


app = Flask(__name__)

def resize_images(test_image_folder, target_size=(224, 224)):
    images = []
    image_files = os.listdir(test_image_folder)
    for image_file in image_files:
        img_path = os.path.join(test_image_folder, image_file)
        img = Image.open(img_path)  # Open image using Pillow
        img_resized = img.resize(target_size)  # Resize to 224x224 using Pillow
        img_resized = np.array(img_resized)  # Convert to numpy array
        images.append(img_resized)
    return np.array(images)

def evaluate_model_on_test_data(model, test_image_folder, test_excel_file):
    # Load and resize images to the expected input size
    test_images = resize_images(test_image_folder)
    
    # Load and preprocess data from the Excel file
    df = pd.read_excel(test_excel_file)
    regions = df["REGION"]
    
    # Make predictions
    predicted_percentages = model.predict(test_images)
    predicted_percentages = np.clip(predicted_percentages, 0, 100)  # Ensure predictions are within 0-100%
    
    print(int(predicted_percentages[0]))
    return int(predicted_percentages[0])


# Define routes after clearing
@app.route('/predict_cyclone1', methods=['POST'])
def predict_cyclone():
    test_image_folder = "test_images"
    test_excel_file = "new_test_data.xlsx"
    df = pd.read_excel(test_excel_file)
    regions = df["REGION"]
    res = []
    
    for region in regions:
        model_path = f"D:/models/{region}_cyclone_prediction_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Evaluating model for region: {region}")
            pr = evaluate_model_on_test_data(model, test_image_folder, test_excel_file)
            res.append({'region': region, 'cyc': pr})
        else:
            res.append({'region': region, 'cyc': None})

    return render_template('result.html', results=res)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

