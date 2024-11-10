import os
import zipfile
import numpy as np
import pandas as pd
from io import BytesIO
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from PIL import Image as PILImage

# Step 1: Extract images from Excel file (for training or testing)
def extract_images_from_excel(file_path, output_folder):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        tabular_data = pd.read_excel(file_path)
        print(f"Extracting images to {output_folder}...")

        image_index = 0
        for file_name in zip_ref.namelist():
            if file_name.startswith('xl/media/') and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_data = zip_ref.read(file_name)
                image = PILImage.open(BytesIO(img_data))

                if image_index < len(tabular_data):
                    region = tabular_data.iloc[image_index]['REGION']
                    cyclone_name = tabular_data.iloc[image_index]['CYCLONE_NAME']
                    new_file_name = f"{region}_{cyclone_name}_{file_name.split('/')[-1]}"
                    new_file_path = os.path.join(output_folder, new_file_name)
                    image.save(new_file_path)
                image_index += 1
        print(f"Extraction completed.")

def extract_images_from_excel_for_test(file_path, output_folder):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        tabular_data = pd.read_excel(file_path)
        print(f"Extracting images to {output_folder}...")

        image_index = 0
        for file_name in zip_ref.namelist():
            if file_name.startswith('xl/media/') and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_data = zip_ref.read(file_name)
                image = PILImage.open(BytesIO(img_data))

                if image_index < len(tabular_data):
                    region = tabular_data.iloc[image_index]['REGION']
                    #cyclone_name = tabular_data.iloc[image_index]['CYCLONE_NAME']
                    new_file_name = f"{region}_test_image.jpg"
                    new_file_path = os.path.join(output_folder, new_file_name)
                    image.save(new_file_path)
                image_index += 1
        print(f"Extraction completed.")
        
# Step 2: Preprocess all images (resize to target size and normalize)
def preprocess_all_images(image_folder, region, target_size=(224, 224)):
    image_data = []
    image_names = []
    count = 0
    for file_name in os.listdir(image_folder):
        region_name = file_name.split("_")[0]
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and region_name == region:
            img_path = os.path.join(image_folder, file_name)
            img = image.load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            image_data.append(img_array)
            image_names.append(file_name)
            count += 1
    print("Images for this region:", count)
    image_data = np.vstack(image_data) if image_data else None
    return image_data, image_names

# Step 3: Load and preprocess tabular data (train or test data)
def load_tabular_data(excel_file, is_train=True):
    df = pd.read_excel(excel_file)
    if is_train:
        max_wind_speed = df['WIND_SPEED(kmph)'].max()
        df['CYCLONE_LIKELIHOOD'] = 100  # Scale to percentage (0-100)
    else:
        df['CYCLONE_LIKELIHOOD'] = None

    # Feature scaling for numerical columns
    scaler = MinMaxScaler()
    df[['WIND_SPEED(kmph)', 'SEA_TEMP(celcius)']] = scaler.fit_transform(df[['WIND_SPEED(kmph)', 'SEA_TEMP(celcius)']])
    
    return df, scaler

# Step 4: Prepare data for training (combine image data and tabular data)
def prepare_data_for_training(image_folder, excel_file, region):
    image_data, image_names = preprocess_all_images(image_folder, region)
    if image_data is None:
        return None, None
    tabular_data, scaler = load_tabular_data(excel_file, is_train=True)
    image_labels = []
    for img_name in image_names:
        img_region = img_name.split('_')[0]
        matching_row = tabular_data[tabular_data['REGION'] == img_region]
        if not matching_row.empty:
            cyclone_name = img_name.split('_')[1].replace('.jpg', '')
            matching_row = matching_row[matching_row['CYCLONE_NAME'] == cyclone_name]
            if not matching_row.empty:
                image_labels.append(matching_row['CYCLONE_LIKELIHOOD'].values[0])
    image_labels = np.array(image_labels)
    return image_data, image_labels, scaler

# Step 5: Build the Model (using pre-trained ResNet50 for feature extraction)
def build_model(input_shape=(224, 224, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the pre-trained layers

    model_input = Input(shape=input_shape)
    x = base_model(model_input)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='linear')(x)  # Linear output for regression

    model = Model(inputs=model_input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
    return model

# Step 6: Train the model on the entire dataset (no split for validation)
def train_and_evaluate_model(image_folder, excel_file, region):
    # Prepare training data
    images, labels, scaler = prepare_data_for_training(image_folder, excel_file, region)
    if images is None or labels is None:
        print(f"Error: Training data preparation failed for region {region}.")
        return

    # Build the model
    model = build_model()

    # Train the model on the entire dataset (no validation split)
    model.fit(images, labels, epochs=15, batch_size=32)

    # Save the trained model
    model_path = f"D:/models/{region}_cyclone_prediction_model.h5"
    model.save(model_path)
    print(f"Model for {region} trained and saved at {model_path}.")

# Step 7: Extract and preprocess images from the test dataset
def extract_and_preprocess_test_data(test_excel_file, test_image_folder):
    extract_images_from_excel_for_test(test_excel_file, test_image_folder)
    df = pd.read_excel(test_excel_file)
    test_images, _ = preprocess_all_images(test_image_folder, region=df["REGION"][0])
    test_data, scaler = load_tabular_data(test_excel_file, is_train=False)
    return test_images, test_data, scaler

# Step 8: Evaluate on the same dataset (train and test without split)
def evaluate_model_on_test_data(model, test_image_folder, test_excel_file):
    test_images, test_data, scaler = extract_and_preprocess_test_data(test_excel_file, test_image_folder)
    predicted_percentages = model.predict(test_images)  # Direct prediction as percentage
    predicted_percentages = np.clip(predicted_percentages, 0, 100)  # Ensure within 0-100 range
    for i in range(len(predicted_percentages)):
        print(f"Predicted: {predicted_percentages[i][0]:.2f}%")
    test_data['Predicted_CYCLONE_LIKELIHOOD'] = predicted_percentages.flatten()
    test_data.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to 'test_predictions.csv'.")

# Example usage
train_image_folder = "train_images"
train_excel_file = "cyclone_new_dataset.xlsx"
extract_images_from_excel(train_excel_file, train_image_folder)
df = pd.read_excel(train_excel_file)
regions = df['REGION'].unique()

for region in regions:
    print(f"Training model for region: {region}")
    train_and_evaluate_model(train_image_folder, train_excel_file, region)

# Step 2: Load the saved model and evaluate it on the test data
test_image_folder = "test_images"
test_excel_file = "test_data.xlsx"
df=pd.read_excel(test_excel_file)
regions=df["REGION"]
for region in regions:
    model_path = f"D:/models/{region}_cyclone_prediction_model.h5"
    #df = pd.read_excel(test_excel_file)
    print(model_path)
    print(df["WIND_SPEED(kmph)"][0])
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Evaluating model for region: {region}")
        evaluate_model_on_test_data(model, test_image_folder, test_excel_file)
    else:
        print(f"No model found for region: {region}")