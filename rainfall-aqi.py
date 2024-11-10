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
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

# Ensure that any previous Flask threads are cleared and run a new one
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
