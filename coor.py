from geojson import Feature, Point, FeatureCollection
import json
import pandas as pd

df = pd.read_csv("city_coordinates.csv")  # Must include City, Latitude, Longitude, predicted_AQI

features = []
for _, row in df.iterrows():
    point = Point((row["Longitude"], row["Latitude"]))  # [lon, lat]
    properties = {
        "City": row["City"],
        "predicted_AQI": row["predicted_AQI"]
    }
    features.append(Feature(geometry=point, properties=properties))

fc = FeatureCollection(features)

with open("india_aqi.geojson", "w") as f:
    f.write(json.dumps(fc))
