import os
import pickle
from flask import Flask, render_template, request
import requests
import numpy as np
import threading

app = Flask(__name__)

# Define the path where models are stored
MODEL_FOLDER = 'AQI_models'

# Latitude and Longitude for each state (example values, replace with accurate data)
state_coordinates = {
    'haryana': {'lat': 30.7333, 'lon': 76.7794},
    'orissa': {'lat': 20.2961, 'lon': 85.8245},
    'rajasthan': {'lat': 26.9124, 'lon': 75.7873},
    'tamilnadu': {'lat': 13.0827, 'lon': 80.2707},
    'arunchal pradesh': {'lat': 27.0844, 'lon': 93.6053},
    'jammu & kashmir': {'lat': 34.0837, 'lon': 74.7973},
    'madhya pradesh': {'lat': 23.2599, 'lon': 77.4126},
    'maharashtra': {'lat': 19.0760, 'lon': 72.8777},
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


@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
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
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    
# Ensure that any previous Flask threads are cleared and run a new one
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
