from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# --- Config ---
pollutants = ['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'CO (mg/m3)', 'SO2 (ug/m3)', 'Ozone (ug/m3)', 'NO2 (ug/m3)', 'NH3 (ug/m3)']

# CPCB breakpoints
cpcb_breakpoints = {
    'PM2.5 (ug/m3)': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    'PM10 (ug/m3)': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    'CO (mg/m3)': [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    'SO2 (ug/m3)': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500)],
    'Ozone (ug/m3)': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    'NO2 (ug/m3)': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500)],
    'NH3 (ug/m3)': [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 3000, 401, 500)]
}

# AQI Categories
aqi_categories = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe")
]

# --- Utilities ---
def calculate_individual_aqi(value, pollutant):
    for bp_low, bp_high, aqi_low, aqi_high in cpcb_breakpoints[pollutant]:
        if bp_low <= value <= bp_high:
            return round(((aqi_high - aqi_low)/(bp_high - bp_low)) * (value - bp_low) + aqi_low)
    return None

def get_aqi_category(aqi_value):
    for low, high, category in aqi_categories:
        if low <= aqi_value <= high:
            return category
    return "Unknown"

def load_model_for_city(city):
    model_path = f'models/{city}.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for city '{city}' not found.")
    return joblib.load(model_path)

def load_data_for_city(city):
    data_path = f'datasets/{city}.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data for city '{city}' not found.")
    # Parse datetime index properly
    return pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

def create_future_dataset_on_request(df, predict_date):
    future_index = pd.date_range(start=f"{predict_date} 00:00", end=f"{predict_date} 23:00", freq='H')
    
    future_df = pd.DataFrame(index=future_index)
    future_df['hour'] = future_df.index.hour
    future_df['dayofmonth'] = future_df.index.day
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['weekofyear'] = future_df.index.isocalendar().week
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter
    future_df['year'] = future_df.index.year

    for pollutant in pollutants:
        future_df[f'{pollutant}_lag_1Y'] = df[pollutant].reindex(future_df.index - pd.Timedelta(days=365))
        future_df[f'{pollutant}_lag_2Y'] = df[pollutant].reindex(future_df.index - pd.Timedelta(days=730))

    # Fill missing values
    future_df = future_df.fillna(method='ffill').fillna(method='bfill')

    return future_df

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    city = data.get('city')
    predict_date = data.get('date')

    if not city or not predict_date:
        return jsonify({"error": "City or Date missing in request!"}), 400

    try:
        # Load model and dataset
        model = load_model_for_city(city)
        df = load_data_for_city(city)

        # Create future dataset
        X_future = create_future_dataset_on_request(df, predict_date)
        # Prepare future input


        # 🔥 Reorder based on model's training feature names
        f_names = model.get_booster().feature_names
        X_future = X_future[f_names]

        # Predict
        preds = model.predict(X_future)
        preds = preds.mean(axis=0)  # Average for the whole day (24 hr)

        predicted_pollutants = dict(zip(pollutants, preds))

        # Calculate AQI for each pollutant
        pollutant_aqi = {}
        for pollutant, value in predicted_pollutants.items():
            aqi = calculate_individual_aqi(value, pollutant)
            if aqi is not None:
                pollutant_aqi[pollutant] = aqi

        # Find dominant pollutant
        if not pollutant_aqi:
            return jsonify({"error": "AQI calculation failed!"}), 500

        dominant_pollutant = max(pollutant_aqi, key=pollutant_aqi.get)
        max_aqi_value = pollutant_aqi[dominant_pollutant]

        return jsonify({
            "city": city,
            "date": predict_date,
            "predicted_AQI": int(max_aqi_value),
            "category": get_aqi_category(max_aqi_value),
            "dominant_pollutant": dominant_pollutant
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
