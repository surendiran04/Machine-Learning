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
