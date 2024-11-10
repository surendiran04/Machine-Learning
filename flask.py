from flask import Flask, render_template_string, request
import threading

app = Flask(_name_)

@app.route('/', methods=['GET', 'POST'])
def home():
    # Function to return 'kk' instead of any user input
    def cap(i_region_name):
        import requests
        from datetime import datetime
        import pandas as pd
        import numpy as np
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
        
        i_city = dict_reg[i_region_name]
        API = '2f44aef9a2277fe209f75cfbaa66025a'  # Replace with your actual API key
        CITY = i_city  # Replace with your city name
        
        BASE_FORECAST = "http://api.openweathermap.org/data/2.5/forecast?"
        url_forecast = BASE_FORECAST + 'appid=' + API + '&q=' + CITY
        
        def get_weather_forecast(url):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check if the request was successful (status code 200)
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
                temperature = entry['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
                pressure = entry['main']['pressure']
                humidity = entry['main']['humidity']
                cloud_coverage = entry['clouds']['all']
                
                rain = entry.get('rain', {}).get('3h', 0)  # Rainfall in the last 3 hours (in mm)
        
                forecast_data.append({
                    'date': date,
                    'Average_Temperature': temperature,
                    'Pressure': pressure,
                    'Estimated_Humidity': humidity,
                    'Cloud_Coverage': cloud_coverage,
                    'Rainfall_3h': rain  # Add rainfall data for this 3-hour period
                })
        
            df_forecast = pd.DataFrame(forecast_data)
            
            daily_forecast = df_forecast.groupby('date').agg({
                'Average_Temperature': 'mean',  # Mean of the temperature
                'Pressure': 'mean',             # Mean of the pressure
                'Estimated_Humidity': 'mean',   # Mean of the humidity
                'Cloud_Coverage': 'mean',       # Mean of the cloud coverage
                'Rainfall_3h': 'sum'           # Sum of the rainfall for each day
            }).head(7)  # Get averages and sums for the next 7 days
        
            total_rainfall = daily_forecast['Rainfall_3h'].sum()
            avg_temperature = daily_forecast['Average_Temperature'].mean()
            avg_pressure = daily_forecast['Pressure'].mean()
            avg_humidity = daily_forecast['Estimated_Humidity'].mean()
            avg_cloud_coverage = daily_forecast['Cloud_Coverage'].mean()
        
            print("\nTotal Rainfall and Averages for the Next 7 Days:")
            print(f"Total Rainfall: {total_rainfall} mm")
            print(f"Average Temperature: {avg_temperature:.2f} °C")
            print(f"Average Pressure: {avg_pressure:.2f} hPa")
            print(f"Average Humidity: {avg_humidity:.2f} %")
            print(f"Average Cloud Coverage: {avg_cloud_coverage:.2f} %")
        else:
            print("Error: Couldn't retrieve forecast data.")
        
        ###
        import joblib
        import random
        
        def calculate_svp(temperature):
            return 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
            
        def load_region_model(region_name):
            model_path = f'XGBMFR/model_{region_name} WRT.pkl'  # Correct path with space handling
            try:
                model = joblib.load(model_path)  # Load the trained model
                print(f"Model for {region_name} loaded successfully.")
                return model
            except FileNotFoundError:
                print(f"Model for {region_name} not found.")
                return None
        
        region_name = i_region_name
        region_model = load_region_model(region_name)
        
        if region_model:
            new_region_data = pd.DataFrame({
                'Altitude': [350.32],
                'Average_Temperature': [avg_temperature],
                'Pressure': [avg_pressure],
                'SVP': [calculate_svp(avg_temperature)],
                'Estimated_Humidity': [avg_humidity],
                'Cloud_Coverage': [avg_cloud_coverage]
            })
        
            prediction = region_model.predict(new_region_data)
            if new_region_data['Average_Temperature'][0] > 32.5:
                prediction[0] *= 0.15
            if prediction[0] < 0:
                prediction[0] = random.uniform(0, 1)
            else:
                prediction[0] *= 0.1
            print(f"Predicted Rainfall for {region_name}: {prediction[0]} mm")

        lst = {"ACTUAL RAINFALL" : total_rainfall, "AVG TEMPERATUE" : avg_temperature, "AVG PRESSURE" : avg_pressure, "AVG SVP" : calculate_svp(avg_temperature), "AVG HUMIDITY" :avg_humidity, "AVG CLOUD COVERAGE" :avg_cloud_coverage, "PREDICTED RAINFALL" :prediction[0]}
        rlst = [{x : float(lst[x]) for x in lst}]
        return rlst
        
    result = ''
    if request.method == 'POST':
        user_input = request.form['text_input']
        result = cap(user_input)  # Call the cap function which always returns 'kk'

    return render_template_string("""
        <html>
            <body>
                <h1>Simple Form</h1>
                <form method="POST">
                    <label for="text_input">Enter text:</label>
                    <input type="text" id="text_input" name="text_input" value="{{ request.form.get('text_input', '') }}">
                    <button type="submit">Submit</button>
                </form>
                {% if result_lines %}
                    <h2>Result:</h2>
                    <ul>
                        {% for key, value in result_lines[0].items() %}
                            <li>{{ key, value }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
            </body>
        </html>
    """, result_lines=result)

def run_flask():
    app.run(debug=True, use_reloader=False)

# Ensure that any previous Flask threads are cleared and run a new one
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
