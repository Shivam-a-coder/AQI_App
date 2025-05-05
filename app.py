from flask import Flask, json, request, jsonify, render_template,send_file
import pandas as pd
import joblib
import os
import logging
import plotly.graph_objects as go
import plotly.io as pio
from flask_cors import CORS
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import json
from flask import Response
from plotly.utils import PlotlyJSONEncoder
import plotly.utils

app = Flask(__name__)
CORS(app)

# --- Logging ---
logging.basicConfig(level=logging.INFO)

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

aqi_categories = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe")
]

# --- Utility Functions ---
def calculate_individual_aqi(value, pollutant):
    for bp_low, bp_high, aqi_low, aqi_high in cpcb_breakpoints[pollutant]:
        if bp_low <= value <= bp_high:
            return round(((aqi_high - aqi_low) / (bp_high - bp_low)) * (value - bp_low) + aqi_low)
    return None

def get_aqi_category(aqi_value):
    for low, high, category in aqi_categories:
        if low <= aqi_value <= high:
            return category
    return "Unknown"

def load_model_for_city(city):
    model_path = f'models/{city}.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for city '{city}' not found at {model_path}")
    return joblib.load(model_path)

def load_data_for_city(city):
    data_path = f'datasets/{city}.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data for city '{city}' not found at {data_path}")
    return pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
import csv

def get_city_coordinates(city_name, csv_path='city_coordinates.csv'):
    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['City'].strip().lower() == city_name.strip().lower():
                    return float(row['Latitude']), float(row['Longitude'])
    except Exception as e:
        logging.warning(f"Could not load coordinates for city {city_name}: {e}")
    return None, None

def create_future_dataset_on_request(df, predict_date):
    future_index = pd.date_range(start=f"{predict_date} 00:00", end=f"{predict_date} 23:00", freq='H')
    future_df = pd.DataFrame(index=future_index)
    future_df['hour'] = future_df.index.hour
    future_df['dayofmonth'] = future_df.index.day
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['weekofyear'] = future_df.index.isocalendar().week.astype(int)
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter
    future_df['year'] = future_df.index.year

    for pollutant in pollutants:
        future_df[f'{pollutant}_lag_1Y'] = df[pollutant].reindex(future_df.index - pd.Timedelta(days=365))
        future_df[f'{pollutant}_lag_2Y'] = df[pollutant].reindex(future_df.index - pd.Timedelta(days=730))

    return future_df.fillna(method='ffill').fillna(method='bfill')




# --- Routes ---
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
        model = load_model_for_city(city)
        df = load_data_for_city(city)
        X_future = create_future_dataset_on_request(df, predict_date)

        f_names = model.get_booster().feature_names
        missing_features = [f for f in f_names if f not in X_future.columns]
        if missing_features:
            return jsonify({"error": f"Missing features in future dataset: {missing_features}"}), 500

        X_future = X_future[f_names]
        preds = model.predict(X_future)
        preds = preds.mean(axis=0)

        predicted_pollutants = dict(zip(pollutants, preds))

        pollutant_aqi = {}
        for pollutant, value in predicted_pollutants.items():
            aqi = calculate_individual_aqi(value, pollutant)
            if aqi is not None:
                pollutant_aqi[pollutant] = aqi

        if not pollutant_aqi:
            return jsonify({"error": "AQI calculation failed!"}), 500

        dominant_pollutant = max(pollutant_aqi, key=pollutant_aqi.get)
        max_aqi_value = pollutant_aqi[dominant_pollutant]
        lat, lon = get_city_coordinates(city)
        if lat is None or lon is None:
            return jsonify({"error": f"Coordinates for city '{city}' not found in stations_info.csv"}), 404

        return jsonify({
            "city": city,
            "date": predict_date,
            "predicted_AQI": int(max_aqi_value),
            "category": get_aqi_category(max_aqi_value),
            "dominant_pollutant": dominant_pollutant,
            "pollutant_aqi": pollutant_aqi,
            "lat": lat,
            "lon": lon
        })


    except FileNotFoundError as fnf_err:
        logging.error(str(fnf_err))
        return jsonify({"error": str(fnf_err)}), 404
    except Exception as e:
        logging.exception("Unhandled error during forecast")
        return jsonify({"error": str(e)}), 500
@app.route('/india_aqi')
def india_aqi():
    predict_date = request.args.get('date')
    if not predict_date:
        return jsonify({"error": "Missing date parameter."}), 400

    try:
        geojson_path = 'static/india_aqi.geojson'
        if not os.path.exists(geojson_path):
            return jsonify({"error": "GeoJSON file not found."}), 404

        with open(geojson_path, 'r') as file:
            geojson_data = json.load(file)

        # Modify properties of each feature by fetching AQI for its city
        for feature in geojson_data.get('features', []):
            city_name = feature['properties'].get('City')
            try:
                model = load_model_for_city(city_name)
                df = load_data_for_city(city_name)
                X_future = create_future_dataset_on_request(df, predict_date)
                f_names = model.get_booster().feature_names
                X_future = X_future[f_names]
                preds = model.predict(X_future).mean(axis=0)
                predicted_pollutants = dict(zip(pollutants, preds))

                pollutant_aqi = {
                    p: calculate_individual_aqi(v, p) for p, v in predicted_pollutants.items()
                    if calculate_individual_aqi(v, p) is not None
                }

                if pollutant_aqi:
                    dom_pollutant = max(pollutant_aqi, key=pollutant_aqi.get)
                    max_aqi = pollutant_aqi[dom_pollutant]
                    feature['properties']['aqi'] = int(max_aqi)
                    feature['properties']['category'] = get_aqi_category(max_aqi)
                    feature['properties']['dominant_pollutant'] = dom_pollutant

            except Exception as e:
                logging.warning(f"Could not predict AQI for {city_name}: {e}")
                feature['properties']['aqi'] = None
                feature['properties']['category'] = "Unavailable"

        return jsonify(geojson_data)

    except Exception as e:
        logging.exception("Error in /india_aqi")
        return jsonify({"error": str(e)}), 500



def load_historical_for_city(city):
    path = os.path.join('datasets', f'{city}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No raw file for {city}")
    
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    return df.sort_index()
@app.route('/historical_grouped')
def historical_grouped():
    city = request.args.get('city')
    if not city:
        return jsonify({'error': 'No city provided.'}), 400

    try:
        df = load_historical_for_city(city)
        if df.empty:
            return jsonify({'error': f"No data found for {city}."}), 404

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        available_pollutants = [col for col in df.columns if col.lower() not in ['latitude', 'longitude']]

        if not available_pollutants:
            return jsonify({'error': f"No pollutant data found for {city}."}), 404

        plots = {}

        for pollutant in available_pollutants:
            fig = go.Figure()

            try:
                # Daily
                daily_data = df[pollutant].resample('1D').mean().ffill().bfill()
                fig.add_trace(go.Scatter(
                    x=daily_data.index.to_list(),
                    y=daily_data.values.tolist(),
                    mode='lines',
                    name='Daily Avg',
                    line=dict(width=2)
                ))

                # Monthly
                monthly_data = df[pollutant].resample('M').mean().ffill().bfill()
                fig.add_trace(go.Scatter(
                    x=monthly_data.index.to_list(),
                    y=monthly_data.values.tolist(),
                    mode='lines',
                    name='Monthly Avg',
                    line=dict(width=2, dash='dash')
                ))

                # Yearly
                yearly_data = df[pollutant].resample('Y').mean().ffill().bfill()
                fig.add_trace(go.Scatter(
                    x=yearly_data.index.to_list(),
                    y=yearly_data.values.tolist(),
                    mode='lines+markers',
                    name='Yearly Avg',
                    line=dict(width=2, dash='dot')
                ))

                fig.update_layout(
                    title=f"{pollutant} Trends in {city} (Daily, Monthly, Yearly)",
                    xaxis_title="Date",
                    yaxis_title=f"{pollutant} Concentration",
                    height=350,  # ← taller plot helps space things out
                    margin=dict(t=80, b=80, l=80, r=120),  # ← increase bottom margin
                    legend=dict(
                        x=1.02, y=1,
                        xanchor='left'
                    ),
                    showlegend=True
                )




                plots[pollutant] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

            except Exception as e:
                logging.warning(f"Failed to process {pollutant}: {str(e)}")
                continue

        return jsonify({
            'city': city,
            'plots': plots,
            'status': 'success'
        })

    except Exception as e:
        logging.exception(f"Error processing historical data for {city}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
