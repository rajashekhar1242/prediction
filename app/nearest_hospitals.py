import json
import numpy as np
from math import sqrt

# Load hospital data once when the module is imported
with open("kaggle_dataset/hospitals.json", "r") as file:
    data = json.load(file)

hospitals_data = data["hospitals"]

def find_nearest_hospitals(user_lat, user_lon, top_n=3):
    coords = np.array([[h["latitude"], h["longitude"]] for h in hospitals_data])
    names = [h["name"] for h in hospitals_data]

    distances = np.sqrt((coords[:, 0] - user_lat) ** 2 + (coords[:, 1] - user_lon) ** 2)

    nearest_indices = distances.argsort()[:top_n]

    results = [
        {
            "name": names[i],
            "latitude": coords[i][0],
            "longitude": coords[i][1],
            "distance": round(distances[i] * 111, 2)  # approx conversion to kilometers
        }
        for i in nearest_indices
    ]
    return results
