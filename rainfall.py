import os
import pickle
import requests
import numpy as np
from flask import Flask, render_template, request
from datetime import datetime
import pandas as pd
from math import exp
import threading

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
                prediction = region_model.predict(new_region_data)
                if new_region_data['Average_Temperature'][0] > 32.5:
                    prediction[0] *= 0.15
                if prediction[0] < 0:
                    prediction[0] = random.uniform(0, 1)
                else:
                    prediction[0] *= 0.1
                if region_name='TAMILNADU':
                    prediction[0] *= 0.7
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

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

# Ensure that any previous Flask threads are cleared and run a new one
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
