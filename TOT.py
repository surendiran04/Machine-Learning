import os
import pickle
import requests
import numpy as np
from flask import Flask, render_template, request
from datetime import datetime
import pandas as pd
from math import exp
import threading
import joblib
import random
from PIL import Image  # Import Pillow's Image module
from tensorflow.keras.models import load_model



app = Flask(__name__)

dict_reg = {
    'ANDAMAN AND NICOBAR': 'PORT BLAIR', 'ANDHRA PRADESH': 'VIJAYAWADA',
    'ARUNACHAL PRADESH': 'ITANAGAR', 'ASSAM AND MEGHALAYA': 'GUWAHATI',
    'BIHAR': 'PATNA', 'CHHATTISGARH': 'RAIPUR', 'COASTAL KARNATAKA': 'MANGALORE',
    'EAST MADHYAPRADESH': 'JABALPUR', 'EAST RAJASTHAN': 'JAIPUR',
    'EAST UTTAR PRADESH': 'LUCKNOW', 'GANGETIC WEST BENGAL': 'KOLKATA',
    'GUJARAT_REGION': 'AHMEDABAD', 'HARYANA DELHI AND CHANDIGARH': 'DELHI',
    'HIMACHAL PRADESH': 'SHIMLA', 'JAMMU AND KASHMIR': 'SRINAGAR',
    'JHARKHAND': 'RANCHI', 'KERALA': 'THIRUVANANTHAPURAM', 'KONKAN AND GOA': 'PANAJI',
    'LAKSHADWEEP': 'KAVARATTI', 'MADHYA MAHARASHTRA': 'AURANGABAD', 'MATATHWADA': 'BEED',
    'NMMT': 'KOHIMA', 'NORTH KARNATAKA': 'HUBLI', 'ORISSA': 'BHUBANESHWAR',
    'PUNJAB': 'CHANDIGARH', 'RAYALSEEMA': 'KURNOOL', 'SAURASHTRA AND KUTCH': 'RAJKOT',
    'SHWBS': 'GANGTOK', 'SOUTH KARNATAKA': 'MYSORE', 'TAMILNADU': 'CHENNAI',
    'TELANGANA': 'HYDERABAD', 'UTTARAKHAND': 'DEHRADUN', 'VIDARBHA': 'NAGPUR',
    'WEST_RAJASTHAN': 'JODHPUR', 'WEST_UTTAR_PRADESH': 'MEERUT'
}

# Dummy function to calculate SVP (Saturated Vapor Pressure) based on temperature
def calculate_svp(temperature):
    # Formula for calculating SVP (Saturated Vapor Pressure) in hPa
    return 6.11 * exp((17.27 * temperature) / (temperature + 237.3))

def load_region_model(region_name):
    model_path = f'XGBMFR/model_{region_name} WRT.pkl'  # Correct path with space handling
    try:
        model = joblib.load(model_path)  # Load the trained model
        print(f"Model for {region_name} loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model for {region_name} not found.")
        return None

# Fetch weather forecast (rainfall) data
def get_weather_forecast(city):
    API = '2f44aef9a2277fe209f75cfbaa66025a'  # Replace with your actual API key
    BASE_FORECAST = "http://api.openweathermap.org/data/2.5/forecast?"
    url_forecast = BASE_FORECAST + 'appid=' + API + '&q=' + city
    
    try:
        response = requests.get(url_forecast)
        response.raise_for_status()
        forecast_data = response.json()
        return forecast_data['list'] if 'list' in forecast_data else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# Route for rainfall forecast prediction
@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    region_name = request.form.get('region')
    if region_name:
        city = dict_reg.get(region_name)
        if city:
            forecast_list = get_weather_forecast(city)
            if forecast_list:
                forecast_data = []
                total_rainfall = 0
                total_temperature = 0
                total_pressure = 0
                total_humidity = 0
                total_cloud_coverage = 0

                for entry in forecast_list:
                    date = datetime.fromtimestamp(entry['dt']).date()
                    temperature = entry['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
                    pressure = entry['main']['pressure']
                    humidity = entry['main']['humidity']
                    cloud_coverage = entry['clouds']['all']
                    rain = entry.get('rain', {}).get('3h', 0)

                    forecast_data.append({
                        'date': date,
                        'Rainfall_3h': rain,
                        'Temperature': temperature,
                        'Pressure': pressure,
                        'Humidity': humidity,
                        'Cloud Coverage': cloud_coverage
                    })

                    # Accumulate totals for averages
                    total_rainfall += rain
                    total_temperature += temperature
                    total_pressure += pressure
                    total_humidity += humidity
                    total_cloud_coverage += cloud_coverage

                # Calculate averages
                num_entries = len(forecast_data)
                avg_temperature = total_temperature / num_entries
                avg_pressure = total_pressure / num_entries
                avg_humidity = total_humidity / num_entries
                avg_cloud_coverage = total_cloud_coverage / num_entries

                # Your prediction model (dummy in this case)
                region_model = load_region_model(region_name)
                if region_model:
                    new_region_data = pd.DataFrame({
                'Altitude': [350.32],
                'Average_Temperature': [avg_temperature],
                'Pressure': [avg_pressure],
                'SVP': [calculate_svp(avg_temperature)],
                'Estimated_Humidity': [avg_humidity],
                'Cloud_Coverage': [avg_cloud_coverage]})
                    
                prediction = region_model.predict(new_region_data)
                if new_region_data['Average_Temperature'][0] > 32.5:
                    prediction[0] *= 0.15
                if prediction[0] < 0:
                    prediction[0] = random.uniform(0, 1)
                if region_name=='TAMILNADU':
                    prediction[0] *= 0.7
                else:
                    prediction[0] *= 0.1
                
                print(f"Predicted Rainfall for {region_name}: {prediction[0]} mm")  # Example prediction (this should be replaced with actual model logic)

                # Create result dictionary for rendering
                lst = {
                    "ACTUAL RAINFALL": total_rainfall,
                    "AVG TEMPERATURE": avg_temperature,
                    "AVG PRESSURE": avg_pressure,
                    "AVG SVP": calculate_svp(avg_temperature),
                    "AVG HUMIDITY": avg_humidity,
                    "AVG CLOUD COVERAGE": avg_cloud_coverage,
                    "PREDICTED RAINFALL": prediction[0]
                }

                # Convert result dictionary into a list of dictionaries
                rlst = [{x: float(lst[x]) for x in lst}]
                
                return render_template('index.html', rlst=rlst, region=region_name)

            else:
                return render_template('index.html', error="Failed to fetch rainfall data.")
        else:
            return render_template('index.html', error="Invalid region selected.")
    return render_template('index.html', error="Region is required.")

##################################################### AQI ###########################################
MODEL_FOLDER = 'AQI_models'
API_KEY = '2f44aef9a2277fe209f75cfbaa66025a'
# Latitude and Longitude for each state (example values, replace with accurate data)
state_coordinates = {
    'haryana': {'lat': 30.7333, 'lon': 76.7794},
    'orissa': {'lat': 20.2961, 'lon': 85.8245},
    'rajasthan': {'lat': 26.9124, 'lon': 75.7873},
    'tamilnadu': {'lat': 13.0827, 'lon': 80.2707},
    'arunchal pradesh': {'lat': 27.0844, 'lon': 93.6053},
    'jammu & kashmir': {'lat': 34.0837, 'lon': 74.7973},
    'madhya pradesh': {'lat': 23.2599, 'lon': 77.4126},
    'maharastra': {'lat': 19.0760, 'lon': 72.8777},
    'assam': {'lat': 26.1433, 'lon': 91.7362},
    'gujarat': {'lat': 23.0225, 'lon': 72.5714},
    'naga mani mizo tripura': {'lat': 25.6747, 'lon': 94.1077},  # Kohima
    'chattisgarh': {'lat': 21.2514, 'lon': 81.6296},
    'kerala': {'lat': 8.5241, 'lon': 76.9366},
    'jharkhand': {'lat': 23.3441, 'lon': 85.3096},
    'uttar pradesh': {'lat': 26.8467, 'lon': 80.9462},
    'karnataka': {'lat': 12.9716, 'lon': 77.5946},
    'west bengal': {'lat': 22.5726, 'lon': 88.3639},
    'bihar': {'lat': 25.5941, 'lon': 85.1376},
    'himachal pradesh': {'lat': 31.1048, 'lon': 77.1734},
    'telangana': {'lat': 17.3850, 'lon': 78.4867},
    'punjab': {'lat': 30.7333, 'lon': 76.7794}
}

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Load model from file
def load_model(state):
    model_file_path = os.path.join(MODEL_FOLDER, f'model_{state}.pkl')
    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    return None

# Fetch AQI data from an external API based on lat/lon
def get_aqi_data(lat, lon):
    # Example API URL, replace with the actual API you're using
    api_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Check if the request was successful (status code 200)
        return response.json()  # Return the JSON response if successful
    except requests.exceptions.Timeout:
        print("Error: The request timed out.")
    except requests.exceptions.ConnectionError as e:
        print("Error: Unable to connect to the AQI API service.", e)
    except requests.exceptions.HTTPError as e:
        print("HTTP error occurred:", e)
    except requests.exceptions.RequestException as e:
        print("An error occurred while making the request:", e)

    # Return None if there was any error
    return None
@app.route('/predict_aqi', methods=['POST'])
def predict():
    state = request.form.get('state')
    print(f"Selected state: {state}")  # Log selected state
    if state:
        coords = state_coordinates.get(state.lower())
        print(f"Coordinates for {state}: {coords}")  # Log coordinates
        if coords:
            lat = coords['lat']
            lon = coords['lon']
            aqi_data = get_aqi_data(lat, lon)
            print(f"Data:{aqi_data}")
            if aqi_data:
                # Extract components (pollutants and their values)
                components = aqi_data.get('list', [])[0].get('components', {})
                
                # Find the dominant pollutant (highest value)
                print(f"C:{components}")
                dominant_pollutant = max(components, key=components.get)
                print(f"Dominant:{dominant_pollutant}")
                # Load the model for the selected state
                model = load_model(state.lower())
                if model:
                    pollutant_value = components[dominant_pollutant]
                    pollutant_value_reshaped = np.array([pollutant_value]).reshape(1, -1)
                    print(f"Dominant:{pollutant_value_reshaped}")
                    prediction = model.predict(pollutant_value_reshaped)
                    
                    aqi_category = categorize_aqi(prediction[0])
                    return render_template('index.html', aqi=int(prediction[0]), category=aqi_category, pollutant=dominant_pollutant, value=pollutant_value,state=state)
                    
                else:
                    return render_template('index.html', error="Model not found for the selected state.")
            else:
                return render_template('index.html', error="Failed to fetch AQI data.")
        else:
            return render_template('index.html', error="Invalid state selected.")
    return render_template('index.html', error="State is required.")

########################################################### Cyclone ##############################################33

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

    print(res)

    return render_template('index.html', results=res)
###FLOOD###
@app.route('/predict_flood', methods=['POST'])
def predict_flood():
    region_name = request.form.get('region')
    if region_name:
        city = dict_reg.get(region_name)
        import requests
                from datetime import datetime
                import pandas as pd
                
                dict_reg = {
                    'ANDAMAN AND NICOBAR': 'PORT BLAIR', 'ANDHRA PRADESH': 'VIJAYAWADA',
                    'ARUNACHAL PRADESH': 'ITANAGAR', 'ASSAM AND MEGHALAYA': 'GUWAHATI',
                    'BIHAR': 'PATNA', 'CHHATTISGARH': 'RAIPUR', 'COASTAL KARNATAKA': 'MANGALORE',
                    'EAST MADHYAPRADESH': 'JABALPUR', 'EAST RAJASTHAN': 'JAIPUR',
                    'EAST UTTAR PRADESH': 'LUCKNOW', 'GANGETIC WEST BENGAL': 'KOLKATA',
                    'GUJARAT_REGION': 'AHMEDABAD', 'HARYANA DELHI AND CHANDIGARH': 'DELHI',
                    'HIMACHAL PRADESH': 'SHIMLA', 'JAMMU AND KASHMIR': 'SRINAGAR',
                    'JHARKHAND': 'RANCHI', 'KERALA': 'THIRUVANANTHAPURAM', 'KONKAN AND GOA': 'PANAJI',
                    'LAKSHADWEEP': 'KAVARATTI', 'MADHYA MAHARASHTRA': 'AURANGABAD', 'MATATHWADA': 'BEED',
                    'NMMT': 'KOHIMA', 'NORTH KARNATAKA': 'HUBLI', 'ORISSA': 'BHUBANESHWAR',
                    'PUNJAB': 'CHANDIGARH', 'RAYALSEEMA': 'KURNOOL', 'SAURASHTRA AND KUTCH': 'RAJKOT',
                    'SHWBS': 'GANGTOK', 'SOUTH KARNATAKA': 'MYSORE', 'TAMILNADU': 'CHENNAI',
                    'TELANGANA': 'HYDERABAD', 'UTTARAKHAND': 'DEHRADUN', 'VIDARBHA': 'NAGPUR',
                    'WEST_RAJASTHAN': 'JODHPUR', 'WEST_UTTAR_PRADESH': 'MEERUT'
                }
                
                # Mock dictionary for river encoding per region
                region_rivers = {'ANDAMAN AND NICOBAR': {'Little Andaman': {'Discharge': 100,
                   'Capacity': 5000},
                  'Andaman': {'Discharge': 150, 'Capacity': 7000},
                  'Nicobar': {'Discharge': 50, 'Capacity': 3000},
                  'Chidiya Tapu': {'Discharge': 30, 'Capacity': 1500},
                  "Ritchie's Archipelago": {'Discharge': 40, 'Capacity': 2000}},
                 'ARUNACHAL PRADESH': {'Brahmaputra': {'Discharge': 15000, 'Capacity': 500000},
                  'Siang': {'Discharge': 12000, 'Capacity': 450000},
                  'Subansiri': {'Discharge': 8000, 'Capacity': 300000},
                  'Kameng': {'Discharge': 5000, 'Capacity': 200000},
                  'Dibang': {'Discharge': 3000, 'Capacity': 150000}},
                 'ASSAM AND MEGHALAYA': {'Brahmaputra': {'Discharge': 12000,
                   'Capacity': 450000},
                  'Barak': {'Discharge': 5000, 'Capacity': 200000},
                  'Dikhow': {'Discharge': 2000, 'Capacity': 80000},
                  'Lohit': {'Discharge': 4000, 'Capacity': 150000},
                  'Rangia': {'Discharge': 1500, 'Capacity': 60000}},
                 'BIHAR': {'Ganga': {'Discharge': 20000, 'Capacity': 750000},
                  'Kosi': {'Discharge': 5000, 'Capacity': 200000},
                  'Sone': {'Discharge': 4000, 'Capacity': 150000},
                  'Bagmati': {'Discharge': 3000, 'Capacity': 120000},
                  'Mahananda': {'Discharge': 2000, 'Capacity': 80000}},
                 'CHHATTISGARH': {'Mahanadi': {'Discharge': 15000, 'Capacity': 600000},
                  'Shahdol': {'Discharge': 1000, 'Capacity': 40000},
                  'Hasdeo': {'Discharge': 2000, 'Capacity': 80000},
                  'Tungabhadra': {'Discharge': 3000, 'Capacity': 120000},
                  'Kharun': {'Discharge': 1000, 'Capacity': 50000}},
                 'ANDHRA PRADESH': {'Godavari': {'Discharge': 15000, 'Capacity': 600000},
                  'Krishna': {'Discharge': 8000, 'Capacity': 300000},
                  'Tungabhadra': {'Discharge': 4000, 'Capacity': 150000},
                  'Penna': {'Discharge': 1500, 'Capacity': 60000},
                  'Nagavali': {'Discharge': 1000, 'Capacity': 40000}},
                 'COASTAL KARNATAKA': {'Sharavati': {'Discharge': 4000, 'Capacity': 150000},
                  'Netravati': {'Discharge': 3000, 'Capacity': 120000},
                  'Kundapura': {'Discharge': 1000, 'Capacity': 40000},
                  'Aghanashini': {'Discharge': 1200, 'Capacity': 45000},
                  'Swarna': {'Discharge': 1500, 'Capacity': 60000}},
                 'EAST MADHYA PRADESH': {'Narmada': {'Discharge': 8000, 'Capacity': 300000},
                  'Mahi': {'Discharge': 3000, 'Capacity': 120000},
                  'Tungabhadra': {'Discharge': 2500, 'Capacity': 100000},
                  'Kali Sindh': {'Discharge': 2000, 'Capacity': 80000},
                  'Chambal': {'Discharge': 1500, 'Capacity': 60000}},
                 'EAST RAJASTHAN': {'Mahi': {'Discharge': 5000, 'Capacity': 200000},
                  'Sabarmati': {'Discharge': 3000, 'Capacity': 120000},
                  'Chambal': {'Discharge': 4000, 'Capacity': 150000},
                  'Banas': {'Discharge': 2000, 'Capacity': 80000},
                  'Luni': {'Discharge': 1000, 'Capacity': 40000}},
                 'EAST UTTAR PRADESH': {'Ganga': {'Discharge': 18000, 'Capacity': 700000},
                  'Yamuna': {'Discharge': 10000, 'Capacity': 400000},
                  'Rapti': {'Discharge': 2000, 'Capacity': 80000},
                  'Ghaghara': {'Discharge': 3000, 'Capacity': 120000},
                  'Sarasvati': {'Discharge': 1500, 'Capacity': 60000}},
                 'GANGETIC WEST BENGAL': {'Hooghly': {'Discharge': 12000, 'Capacity': 450000},
                  'Damodar': {'Discharge': 8000, 'Capacity': 300000},
                  'Sundarbans': {'Discharge': 5000, 'Capacity': 200000},
                  'Teesta': {'Discharge': 2000, 'Capacity': 80000},
                  'Subarnarekha': {'Discharge': 3000, 'Capacity': 120000}},
                 'GUJARAT REGION': {'Mahi': {'Discharge': 8000, 'Capacity': 300000},
                  'Sabarmati': {'Discharge': 5000, 'Capacity': 200000},
                  'Narmada': {'Discharge': 12000, 'Capacity': 450000},
                  'Tapi': {'Discharge': 6000, 'Capacity': 250000},
                  'Krishna': {'Discharge': 4000, 'Capacity': 150000}},
                 'HARYANA DELHI CHANDIGARH': {'Yamuna': {'Discharge': 10000,
                   'Capacity': 400000},
                  'Ghaggar': {'Discharge': 4000, 'Capacity': 150000},
                  'Sutlej': {'Discharge': 5000, 'Capacity': 200000},
                  'Markanda': {'Discharge': 1500, 'Capacity': 60000},
                  'Chambal': {'Discharge': 2000, 'Capacity': 80000}},
                 'HIMACHAL PRADESH': {'Beas': {'Discharge': 6000, 'Capacity': 200000},
                  'Chenab': {'Discharge': 4000, 'Capacity': 150000},
                  'Ravi': {'Discharge': 2500, 'Capacity': 100000},
                  'Sutlej': {'Discharge': 5000, 'Capacity': 200000},
                  'Jhelum': {'Discharge': 3000, 'Capacity': 120000}},
                 'JAMMU AND KASHMIR': {'Jhelum': {'Discharge': 12000, 'Capacity': 450000},
                  'Chenab': {'Discharge': 8000, 'Capacity': 300000},
                  'Tawi': {'Discharge': 3000, 'Capacity': 120000},
                  'Ravi': {'Discharge': 2500, 'Capacity': 100000},
                  'Sindh': {'Discharge': 2000, 'Capacity': 80000}},
                 'JHARKHAND': {'Damodar': {'Discharge': 8000, 'Capacity': 300000},
                  'Subarnarekha': {'Discharge': 5000, 'Capacity': 200000},
                  'Koel': {'Discharge': 3000, 'Capacity': 120000},
                  'Koyel': {'Discharge': 1000, 'Capacity': 40000},
                  'Kharkai': {'Discharge': 2000, 'Capacity': 80000}},
                 'KERALA': {'Periyar': {'Discharge': 10000, 'Capacity': 400000},
                  'Pamba': {'Discharge': 8000, 'Capacity': 300000},
                  'Bharathapuzha': {'Discharge': 6000, 'Capacity': 200000},
                  'Chaliyar': {'Discharge': 2500, 'Capacity': 100000},
                  'Muvattupuzha': {'Discharge': 3000, 'Capacity': 120000}},
                 'KONKAN AND GOA': {'Mandovi': {'Discharge': 4000, 'Capacity': 150000},
                  'Zuari': {'Discharge': 3000, 'Capacity': 120000},
                  'Ganges': {'Discharge': 2500, 'Capacity': 100000},
                  'Sahyadri': {'Discharge': 2000, 'Capacity': 80000},
                  'Terekhol': {'Discharge': 1500, 'Capacity': 60000}},
                 'LAKSHADWEEP': {'Amini': {'Discharge': 50, 'Capacity': 2000},
                  'Androth': {'Discharge': 40, 'Capacity': 1500},
                  'Agatti': {'Discharge': 60, 'Capacity': 2500},
                  'Kavaratti': {'Discharge': 30, 'Capacity': 1200},
                  'Kalapeni': {'Discharge': 20, 'Capacity': 800}},
                 'MADHYA MAHARASHTRA': {'Godavari': {'Discharge': 12000, 'Capacity': 450000},
                  'Bhima': {'Discharge': 8000, 'Capacity': 300000},
                  'Tungabhadra': {'Discharge': 6000, 'Capacity': 250000},
                  'Krishna': {'Discharge': 5000, 'Capacity': 200000},
                  'Purna': {'Discharge': 1500, 'Capacity': 60000}},
                 'MATATHWADA': {'Godavari': {'Discharge': 10000, 'Capacity': 400000},
                  'Penganga': {'Discharge': 5000, 'Capacity': 200000},
                  'Purna': {'Discharge': 3000, 'Capacity': 120000},
                  'Tungabhadra': {'Discharge': 4000, 'Capacity': 150000},
                  'Sina': {'Discharge': 1000, 'Capacity': 40000}},
                 'NMMT': {'Doyang': {'Discharge': 3000, 'Capacity': 120000},
                  'Zungki': {'Discharge': 1000, 'Capacity': 50000},
                  'Irang': {'Discharge': 2000, 'Capacity': 80000},
                  'Tlawng': {'Discharge': 1500, 'Capacity': 60000},
                  'Barak': {'Discharge': 4000, 'Capacity': 150000}},
                 'NORTH KARNATAKA': {'Krishna': {'Discharge': 7000, 'Capacity': 250000},
                  'Bhima': {'Discharge': 3500, 'Capacity': 130000},
                  'Malaprabha': {'Discharge': 1200, 'Capacity': 40000},
                  'Tungabhadra': {'Discharge': 4500, 'Capacity': 160000},
                  'Ghataprabha': {'Discharge': 1000, 'Capacity': 30000}},
                 'ORISSA': {'Mahanadi': {'Discharge': 9000, 'Capacity': 300000},
                  'Brahmani': {'Discharge': 3000, 'Capacity': 120000},
                  'Subarnarekha': {'Discharge': 2000, 'Capacity': 70000},
                  'Tel': {'Discharge': 1000, 'Capacity': 40000},
                  'Rushikulya': {'Discharge': 1500, 'Capacity': 50000}},
                 'PUNJAB': {'Ravi': {'Discharge': 8000, 'Capacity': 280000},
                  'Beas': {'Discharge': 7000, 'Capacity': 250000},
                  'Sutlej': {'Discharge': 6000, 'Capacity': 230000},
                  'Ghaggar': {'Discharge': 2000, 'Capacity': 70000},
                  'Chandrabhaga': {'Discharge': 1000, 'Capacity': 30000}},
                 'RAYALSEEMA': {'Penna': {'Discharge': 4000, 'Capacity': 150000},
                  'Tungabhadra': {'Discharge': 4500, 'Capacity': 160000},
                  'Chitravathi': {'Discharge': 1000, 'Capacity': 40000},
                  'Kundali': {'Discharge': 1500, 'Capacity': 50000},
                  'Papagni': {'Discharge': 1200, 'Capacity': 45000}},
                 'SAURASHTRA AND KUTCH': {'Sabarmati': {'Discharge': 6000, 'Capacity': 230000},
                  'Mahi': {'Discharge': 4000, 'Capacity': 150000},
                  'Narmada': {'Discharge': 7000, 'Capacity': 270000},
                  'Tapi': {'Discharge': 2000, 'Capacity': 80000},
                  'Shetrunji': {'Discharge': 1000, 'Capacity': 40000}},
                 'SOUTH KARNATAKA': {'Cauvery': {'Discharge': 12000, 'Capacity': 450000},
                  'Kabini': {'Discharge': 3000, 'Capacity': 100000},
                  'Krishna': {'Discharge': 7000, 'Capacity': 250000},
                  'Tungabhadra': {'Discharge': 4500, 'Capacity': 160000},
                  'Hemavati': {'Discharge': 1500, 'Capacity': 50000}},
                 'SHWBS': {'Teesta': {'Discharge': 7000, 'Capacity': 260000},
                  'Mahananda': {'Discharge': 5000, 'Capacity': 180000},
                  'Jaldhaka': {'Discharge': 2000, 'Capacity': 70000},
                  'Sankosh': {'Discharge': 1500, 'Capacity': 60000},
                  'Rangit': {'Discharge': 1000, 'Capacity': 40000}},
                 'TAMILNADU': {'Kaveri': {'Discharge': 12000, 'Capacity': 450000},
                  'Vaigai': {'Discharge': 4000, 'Capacity': 150000},
                  'Palar': {'Discharge': 1000, 'Capacity': 40000},
                  'Tungabhadra': {'Discharge': 4500, 'Capacity': 160000},
                  'Noyyal': {'Discharge': 1500, 'Capacity': 50000}},
                 'TELANGANA': {'Godavari': {'Discharge': 15000, 'Capacity': 500000},
                  'Krishna': {'Discharge': 7000, 'Capacity': 250000},
                  'Musi': {'Discharge': 1000, 'Capacity': 40000},
                  'Pranahita': {'Discharge': 3000, 'Capacity': 120000},
                  'Manjeera': {'Discharge': 1200, 'Capacity': 45000}},
                 'UTTARKHAND': {'Ganga': {'Discharge': 16000, 'Capacity': 600000},
                  'Yamuna': {'Discharge': 7000, 'Capacity': 260000},
                  'Saraswati': {'Discharge': 1000, 'Capacity': 40000},
                  'Bhagirathi': {'Discharge': 5000, 'Capacity': 180000},
                  'Alaknanda': {'Discharge': 3000, 'Capacity': 120000}},
                 'VIDARBHA': {'Wainganga': {'Discharge': 3000, 'Capacity': 110000},
                  'Penganga': {'Discharge': 2000, 'Capacity': 80000},
                  'Pench': {'Discharge': 1500, 'Capacity': 60000},
                  'Kolar': {'Discharge': 1000, 'Capacity': 40000},
                  'Chandrabhaga': {'Discharge': 1200, 'Capacity': 45000}},
                 'WEST_RAJASTHAN': {'Luni': {'Discharge': 5000, 'Capacity': 180000},
                  'Sabarmati': {'Discharge': 6000, 'Capacity': 230000},
                  'Mahi': {'Discharge': 4000, 'Capacity': 150000},
                  'Saraswati': {'Discharge': 1000, 'Capacity': 40000},
                  'Banas': {'Discharge': 1500, 'Capacity': 60000}},
                 'WEST_UTTAR_PRADESH': {'Yamuna': {'Discharge': 7000, 'Capacity': 250000},
                  'Ganga': {'Discharge': 16000, 'Capacity': 600000},
                  'Saraswati': {'Discharge': 1000, 'Capacity': 40000},
                  'Krishna': {'Discharge': 4000, 'Capacity': 150000},
                  'Ganges': {'Discharge': 15000, 'Capacity': 550000}}}
                
                # Define encodings for subdivisions and rivers
                subdivision_encoding = {name: i+1 for i, name in enumerate(dict_reg.keys())}
                river_encoding = {river: j+1 for j, rivers in enumerate(region_rivers.values()) for river in rivers}
                
                # Get the region and API info
                i_region_name = input("ENTER REGION : ")
                i_city = dict_reg.get(i_region_name)
                API = '2f44aef9a2277fe209f75cfbaa66025a'  # Replace with your actual API key
                
                # URL for weather forecast
                BASE_FORECAST = "http://api.openweathermap.org/data/2.5/forecast?"
                url_forecast = BASE_FORECAST + 'appid=' + API + '&q=' + i_city
                
                def get_weather_forecast(url):
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        forecast_data = response.json()
                        
                        if 'list' not in forecast_data:
                            print("Error: Forecast data is missing.")
                            return None
                        
                        return forecast_data['list']
                    
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching data from API: {e}")
                        return None
                
                forecast_list = get_weather_forecast(url_forecast)
                
                if forecast_list:
                    forecast_data = []
                    
                    for entry in forecast_list:
                        date = datetime.fromtimestamp(entry['dt']).date()
                        rain = entry.get('rain', {}).get('3h', 0)
                        forecast_data.append({'date': date, 'Rainfall_3h': rain})
                
                    df_forecast = pd.DataFrame(forecast_data)
                    
                    # Sum of rainfall as discharge (for simplicity)
                    daily_forecast = df_forecast.groupby('date').agg({'Rainfall_3h': 'sum'}).head(7)
                    total_rainfall = daily_forecast['Rainfall_3h'].sum()
                    # Input for region and displaying city, rivers, discharge, and capacity
                    total_discharge = 0
                    total_capacity = 0
                    river_count = 0
                    for river, values in river_data[i_region_name].items():
                        total_discharge += values['Discharge']
                        total_capacity += values['Capacity']
                        river_count += 1
                
                    discharge = total_discharge  # Using cumulative rainfall as discharge
                
                    # Encode subdivision and rivers
                    subdivision = subdivision_encoding.get(i_region_name, 0)
                    rivers = region_rivers.get(i_region_name, [])
                    river_codes = [river_encoding.get(river, 0) for river in rivers]
                    
                    # Current month
                    month = datetime.now().month
                
                    # Prepare the input data
                    input_data = pd.DataFrame({
                        'RAINFALL': [total_rainfall*4 if total_rainfall < 20 else total_rainfall*3],
                        'DISCHARGE': [discharge],
                        'SUBDIVISION': [subdivision],
                        'RIVER': [river_count],  # Use first river's code if available
                        'RAINFALL_DISCHARGE_INTERACTION': [total_rainfall * discharge],
                        'MONTH': [month]
                    })
                
                    print("\nGenerated Input Data for Model:")
                    print(input_data)
                else:
                    print("Error: Couldn't retrieve forecast data.")
                
                import os
                import pandas as pd
                import numpy as np
                import joblib
                from sklearn.preprocessing import StandardScaler
                
                # Function to add noise to the features (same as during training)
                def add_noise_to_features(X, noise_factor=10):
                    numeric_columns = X.select_dtypes(include=[np.number]).columns
                    noise = np.random.normal(8, noise_factor, X[numeric_columns].shape)
                    X_noisy = X.copy()
                    X_noisy[numeric_columns] = X[numeric_columns] + noise
                    return X_noisy
                
                # Define the folder path where models are stored
                model_save_folder = 'XGBFF'  # Folder with the saved models
                # Load the model
                model_filename = os.path.join(model_save_folder, f'model_{i_region_name}.pkl')  # Replace with actual model filename
                model = joblib.load(model_filename)
                
                # Preprocess input data (same steps as during training)
                # Add noise to the input features
                input_data_noisy = add_noise_to_features(input_data)
                
                # Scale the input data (using the scaler used during training)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(input_data_noisy)
                
                # Make the prediction using the loaded model
                prediction = model.predict(X_scaled)
                
                print(f"Prediction: {prediction}")
                return render_template('index.html', rlst=prediction, region=region_name)

            else:
                return render_template('index.html', error="Failed to fetch rainfall data.")
        else:
            return render_template('index.html', error="Invalid region selected.")
    return render_template('index.html', error="Region is required.")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
