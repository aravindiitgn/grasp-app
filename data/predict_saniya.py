import pandas as pd
import numpy as np
import joblib
from itertools import combinations

# Load the trained model and scaler exactly as before
model_path = 'final_workspace_volume_model.pkl'
model = joblib.load(model_path)

scaler_path = 'scaler.pkl'
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    scaler = None
def prepare_input_features(objA_shape, objA_dimensions, objB_shape, objB_dimensions):
    """
    Prepare the input feature vector for the model based on object shapes and dimensions.
    """
    # Calculate object volumes
    def calculate_volume(shape, dimensions):
        if shape == [1, 0, 0]:  # Cuboid
            x, y, z = dimensions
            return 2 * x * 2 * y * 2 * z
        elif shape == [0, 1, 0]:  # Sphere
            r = dimensions[0]
            return (4 / 3) * np.pi * (r) ** 3
        elif shape == [0, 0, 1]:  # Cylinder
            r, h = dimensions
            return np.pi * (r) ** 2 * (2 * h)
        else:
            return 0

    # Calculate grasping length (2 * Y for cuboid/cylinder)
    def calculate_grasping_length(dimensions):
        return 2 * dimensions[1] if len(dimensions) > 1 else 0

    # Calculate the volume and grasping length
    objA_volume = calculate_volume(objA_shape, objA_dimensions)
    objB_volume = calculate_volume(objB_shape, objB_dimensions)
    objA_grasping_length = calculate_grasping_length(objA_dimensions)
    objB_grasping_length = calculate_grasping_length(objB_dimensions)

    # Prepare one-hot encoding for shape
    objA_shape_one_hot = objA_shape
    objB_shape_one_hot = objB_shape

    # Combine all input features into a single list
    input_features = [
        objA_volume, 
        objA_grasping_length, 
        *objA_shape_one_hot, 
        objB_volume, 
        objB_grasping_length, 
        *objB_shape_one_hot
    ]
    return input_features

def predict_workspace_volumes(model, objA, objB):
    """
    Predict the workspace volumes for both grasp sequences:
    1. A first, B second
    2. B first, A second
    """
    # Prepare inputs for case (a) A first, B second
    input_features_a = prepare_input_features(objA['shape'], objA['dimensions'], objB['shape'], objB['dimensions'])

    # Prepare inputs for case (b) B first, A second
    input_features_b = prepare_input_features(objB['shape'], objB['dimensions'], objA['shape'], objA['dimensions'])

    # Convert inputs to DataFrame
    df_a = pd.DataFrame([input_features_a])
    df_b = pd.DataFrame([input_features_b])

    # Normalize the features if a scaler is used
    if scaler:
        X_a_scaled = scaler.transform(df_a)
        X_b_scaled = scaler.transform(df_b)
    else:
        X_a_scaled = df_a
        X_b_scaled = df_b

    # Predict volumes for both cases
    volume_a_first, volume_a_both = model.predict(X_a_scaled)[0]
    volume_b_first, volume_b_both = model.predict(X_b_scaled)[0]

    # Calculate total volumes for each grasp sequence
    total_volume_a = volume_a_first + volume_a_both
    total_volume_b = volume_b_first + volume_b_both

    print(f"Grasp Sequence A -> B: Volume after 1st grasp = {volume_a_first}, Volume after both grasps = {volume_a_both}, Total Volume = {total_volume_a}")
    print(f"Grasp Sequence B -> A: Volume after 1st grasp = {volume_b_first}, Volume after both grasps = {volume_b_both}, Total Volume = {total_volume_b}")

    # Determine the better grasp sequence
    if total_volume_a > total_volume_b:
        print(f"Better grasp sequence is: **A first, B second** with total volume = {total_volume_a}")
    elif total_volume_b > total_volume_a:
        print(f"Better grasp sequence is: **B first, A second** with total volume = {total_volume_b}")
    else:
        print(f"Both grasp sequences have the same total volume: {total_volume_a}")

# Define your five objects in mm
raw_objects = {
    'obj1': {'diameter_mm': 13.8, 'length_mm': 115},
    'obj2': {'diameter_mm': 15.1, 'length_mm': 138.1},
    'obj3': {'diameter_mm': 24.3, 'length_mm': 88.7},
    'obj4': {'diameter_mm': 30.2, 'length_mm': 123.5},
    'obj5': {'diameter_mm': 52.7, 'length_mm': 114.6},
}

# Convert into the format your model expects:
#   shape = [0,0,1] for cylinder
#   dimensions = [radius_m, half_length_m]
objects = {}
for name, dims in raw_objects.items():
    r_m = (dims['diameter_mm'] / 2) / 1000.0
    half_len_m = (dims['length_mm'] / 2) / 1000.0
    objects[name] = {
        'shape': [0, 0, 1],
        'dimensions': [r_m, half_len_m]
    }

# Now loop over all unique pairs and predict
for nameA, nameB in combinations(objects.keys(), 2):
    print(f"\n=== Combination: {nameA} & {nameB} ===")
    predict_workspace_volumes(model, objects[nameA], objects[nameB])
