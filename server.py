from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import json
from PIL import Image
import io
import pickle
import os
import cv2
from threading import Lock

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
DATA_FILE = "training_data.pkl"
MODELS_FILE = "models.pkl"
SCALER_FILE = "scaler.pkl"

# Add lock for thread safety
data_lock = Lock()

# Initialize data
training_data = []
training_labels = []

# Improved model parameters
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean',
    algorithm='auto'
)

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    class_weight='balanced'
)

scaler = StandardScaler()

def extract_shape_features(img_array):
    # Convert to binary
    _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_array.flatten() / 255.0
    
    # Get the largest contour
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calculate shape features
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Get convex hull features
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Append shape features to flattened image
    base_features = img_array.flatten() / 255.0
    shape_features = np.array([circularity, solidity, area/(28*28), perimeter/(4*28)])
    
    return np.concatenate([base_features, shape_features])

def process_image(image_data):
    # Convert to grayscale and process
    img = Image.open(io.BytesIO(image_data)).convert('L')
    img = np.array(img)
    
    # Resize to consistent size
    img = cv2.resize(img, (28, 28))
    
    # Extract features including shape characteristics
    features = extract_shape_features(img)
    return features

@app.post("/train")
async def train_shape(file: UploadFile, label: str):
    global training_data, training_labels
    
    contents = await file.read()
    features = process_image(contents)
    
    with data_lock:
        training_data.append(features)
        training_labels.append(label)
        current_count = len(training_labels)
        
        if current_count > 1:
            X = np.array(training_data)
            y = np.array(training_labels)
            
            # Scale features
            X_scaled = scaler.fit_transform(X)
            
            # Train models
            knn_model.fit(X_scaled, y)
            svm_model.fit(X_scaled, y)
            
            # Save data
            with open(DATA_FILE, 'wb') as f:
                pickle.dump({
                    'features': training_data,
                    'labels': training_labels
                }, f)
            
            with open(MODELS_FILE, 'wb') as f:
                pickle.dump({
                    'knn': knn_model,
                    'svm': svm_model
                }, f)
            
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(scaler, f)
    
    return {
        "status": "success",
        "message": f"Trained as {label}",
        "total_samples": current_count
    }

@app.post("/predict")
async def predict_shape(file: UploadFile):
    if len(training_labels) <= 1:
        return {"prediction": "Need more training data"}
    
    contents = await file.read()
    features = process_image(contents)
    
    with data_lock:
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get predictions
        knn_pred = knn_model.predict(features_scaled)[0]
        knn_prob = np.max(knn_model.predict_proba(features_scaled))
        
        svm_pred = svm_model.predict(features_scaled)[0]
        svm_prob = np.max(svm_model.predict_proba(features_scaled))
        
        prediction_data = {
            "knn_prediction": knn_pred,
            "knn_confidence": float(knn_prob),
            "svm_prediction": svm_pred,
            "svm_confidence": float(svm_prob)
        }
        
        return prediction_data

@app.get("/status")
async def get_status():
    with data_lock:
        return {
            "training_samples": len(training_labels),
            "unique_shapes": list(set(training_labels)),
            "samples_per_shape": {
                shape: training_labels.count(shape)
                for shape in set(training_labels)
            }
        }