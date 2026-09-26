import cv2
import numpy as np
import pandas as pd
from osgeo import gdal
import math


# Step 1: Load inputs

img = cv2.imread("/content/isro image.jpg", cv2.IMREAD_GRAYSCALE)
nasa_catalog = pd.read_csv("/content/crater-data.csv")


# Step 2: Preprocess ISRO image

img_eq = cv2.equalizeHist(img)
img_blur = cv2.GaussianBlur(img_eq, (5,5), 0)


# Step 3: Detect craters

circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2,
                           minDist=40, param1=50, param2=30,
                           minRadius=10, maxRadius=120)

# Step 4: Suppress overlapping detections

def suppress_overlaps(circles, min_dist=20):
    filtered = []
    for (x,y,r) in circles:
        keep = True
        for (fx,fy,fr) in filtered:
            if np.sqrt((x-fx)**2 + (y-fy)**2) < min_dist:
                keep = False
                break
        if keep:
            filtered.append((x,y,r))
    return filtered

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
crater_data = []

if circles is not None:
    circles = np.uint16(np.around(circles[0,:]))
    clean_circles = suppress_overlaps(circles, min_dist=25)

    for idx, (x,y,r) in enumerate(clean_circles, start=1):
        cv2.circle(output, (x,y), r, (0,0,255), 2)   # red rim
        cv2.rectangle(output, (x-2,y-2), (x+2,y+2), (255,0,0), -1) # blue center
        crater_data.append({"id": idx, "x": x, "y": y, "radius_px": r})

cv2.imwrite("isro_crater_overlay_clean.png", output)


# Step 5: Convert pixel coords to lat/long

dataset = gdal.Open("/content/isro image.jpg")
gt = dataset.GetGeoTransform()

def pixel_to_geo(x, y, gt):
    lon = gt[0] + x*gt[1] + y*gt[2]
    lat = gt[3] + x*gt[4] + y*gt[5]
    return lat, lon

for crater in crater_data:
    lat, lon = pixel_to_geo(crater["x"], crater["y"], gt)
    crater["lat"] = lat
    crater["lon"] = lon


# Step 6: Match with NASA catalog

def haversine(lat1, lon1, lat2, lon2):
    R = 1737.4  # Moon radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

for crater in crater_data:
    nasa_catalog["distance"] = nasa_catalog.apply(
        lambda row: haversine(crater["lat"], crater["lon"], row["Latitude"], row["Longitude"]), axis=1)
    match = nasa_catalog.loc[nasa_catalog["distance"].idxmin()]
    crater["nasa_match"] = match["Crater ID"]
    crater["diameter_km"] = 2 * crater["radius_px"] * gt[1] / 1000
    crater["nasa_diameter_km"] = match["Diameter (km)"]
    crater["error_m"] = (crater["diameter_km"] - match["Diameter (km)"]) * 1000

# Step 7: Crater-to-crater distances

for i in range(1, len(crater_data)):
    d = haversine(crater_data[i-1]["lat"], crater_data[i-1]["lon"],
                  crater_data[i]["lat"], crater_data[i]["lon"])
    crater_data[i]["distance_prev_km"] = d
crater_data[0]["distance_prev_km"] = None

# Step 8: Save CSV results

results = pd.DataFrame(crater_data)
results.to_csv("isro_crater_results.csv", index=False)
print(results.head())