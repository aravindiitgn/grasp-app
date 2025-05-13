import pandas as pd
import numpy as np
import joblib
import random

# Load the trained model
model_path = 'final_workspace_volume_model.pkl'  # Path to the saved model
model = joblib.load(model_path)

# Load the MinMaxScaler if it was saved
scaler_path = 'scaler.pkl'  # If the scaler was saved, load it, otherwise reinitialize it
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    scaler = None  # If the scaler was not saved, you may have to reinitialize it

# Function to generate random dimensions for a cube
def generate_cube_dimensions():
    x = np.random.uniform(0.005, 0.03)  # 2*x <= 60, so x is between 0.1 and 30
    y = np.random.uniform(0.005, 0.0625)  # 2*y <= 1.25, so y is between 0.1 and 0.625
    z = x  # x = z
    return [round(x, 3), round(y, 3), round(z, 3)]

# Function to generate random dimensions for a cylinder
def generate_cylinder_dimensions():
    r = np.random.uniform(0.005, 0.03)  # r <= 0.3
    h = np.random.uniform(0.005, 0.0625)  # 2*h <= 1.25, so h is between 0.1 and 0.625
    return [round(r, 3), round(h, 3)]

# Function to generate random dimensions for a sphere
def generate_sphere_dimensions():
    r = np.random.uniform(0.005, 0.03)  # r <= 0.3
    return [round(r, 3)]

# Function to generate random shape and dimensions
def generate_random_object():
    shape_type = random.choice(['Cuboid', 'Sphere', 'Cylinder'])
    if shape_type == 'Cuboid':
        shape = [1, 0, 0]
        dimensions = generate_cube_dimensions()
    elif shape_type == 'Sphere':
        shape = [0, 1, 0]
        dimensions = generate_sphere_dimensions()
    else:  # Cylinder
        shape = [0, 0, 1]
        dimensions = generate_cylinder_dimensions()
    return {'shape': shape, 'dimensions': dimensions}

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

    return total_volume_a, total_volume_b

# Generate and predict for 10 random object pairs
for i in range(10):
    objA = generate_random_object()
    objB = generate_random_object()
    
    total_volume_a, total_volume_b = predict_workspace_volumes(model, objA, objB)
    
    print(f"\nPair {i+1}:")
    print(f"Object A: Shape={objA['shape']}, Dimensions={objA['dimensions']}")
    print(f"Object B: Shape={objB['shape']}, Dimensions={objB['dimensions']}")
    print(f"Grasp Sequence A -> B: Total Volume = {total_volume_a}")
    print(f"Grasp Sequence B -> A: Total Volume = {total_volume_b}")
    
    if total_volume_a > total_volume_b:
        print(f"**Better grasp sequence: A first, B second**")
    elif total_volume_b > total_volume_a:
        print(f"**Better grasp sequence: B first, A second**")
    else:
        print(f"**Both sequences are equal**")
