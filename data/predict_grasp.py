import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_path = 'final_workspace_volume_model.pkl'  # Path to the saved model
model = joblib.load(model_path)

# Load the MinMaxScaler if it was saved
scaler_path = 'scaler.pkl'  # If the scaler was saved, load it, otherwise reinitialize it
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    scaler = None  # If the scaler was not saved, you may have to reinitialize it

# Function to prepare input features for the model
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

# Function to predict the workspace volumes for two grasp sequences
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

# Example objects A and B
object_A = {
    'shape': [1, 0, 0],  # Cuboid
    'dimensions': [0.05, 0.03, 0.05]  # [x, y, z] in half dimensions
}

object_B = {
    'shape': [0, 1, 0],  # Sphere
    'dimensions': [0.03]  # [r] (radius)
}

# Predict the best grasp sequence for objects A and B
predict_workspace_volumes(model, object_A, object_B)
