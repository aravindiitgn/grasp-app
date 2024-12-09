import pandas as pd
import ast
import numpy as np

# File paths
shapes_csv_path = 'random_object_shapes_rounded.csv'
volumes_xml_1_path = 'data/workspace_volumes_xml_1.csv'
volumes_xml_2_path = 'data/workspace_volumes_xml_2.csv'

# Read CSV files
shapes_df = pd.read_csv(shapes_csv_path)
volumes_xml_1_df = pd.read_csv(volumes_xml_1_path)
volumes_xml_2_df = pd.read_csv(volumes_xml_2_path)

# Function to calculate volume for different shapes
def calculate_volume(shape, dimensions):
    """Calculate volume based on shape and dimensions."""
    if shape == [1, 0, 0]:  # Cube
        x, y, z = dimensions
        volume = 2 * x * 2 * y * 2 * z 
    elif shape == [0, 1, 0]:  # Sphere
        r = dimensions[0]
        volume = (4 / 3) * np.pi * (r) ** 3  
    elif shape == [0, 0, 1]:  # Cylinder
        r, h = dimensions
        volume = np.pi * (r) ** 2 * (2 * h)  
    else:
        volume = 0
    return round(volume, 3)

# Extract and process the shapes DataFrame
processed_data = {
    "Object 1 Volume": [],
    "Object 1 Grasping Length": [],
    "Object 1 Shape (Cuboid, Sphere, Cylinder)": [],
    "Object 2 Volume": [],
    "Object 2 Grasping Length": [],
    "Object 2 Shape (Cuboid, Sphere, Cylinder)": [],
    "Remaining Workspace Volume after first grasp": [],
    "Remaining Workspace Volume after both grasps": []
}

for index, row in shapes_df.iterrows():
    # Extract Object 1 details
    obj1_shape = ast.literal_eval(row['Object 1 Shape'])
    obj1_dimensions = ast.literal_eval(row['Obj 1 Dimensions'])
    obj1_volume = calculate_volume(obj1_shape, obj1_dimensions)
    obj1_grasping_length = 2 * obj1_dimensions[1] if len(obj1_dimensions) > 1 else 0  # Length for Y axis
    
    # Extract Object 2 details
    obj2_shape = ast.literal_eval(row['Object 2 Shape'])
    obj2_dimensions = ast.literal_eval(row['Obj 2 Dimensions'])
    obj2_volume = calculate_volume(obj2_shape, obj2_dimensions)
    obj2_grasping_length = 2 * obj2_dimensions[1] if len(obj2_dimensions) > 1 else 0  # Length for Y axis

    # One-hot encoding for shape
    obj1_shape_one_hot = obj1_shape
    obj2_shape_one_hot = obj2_shape
    
    # Random Grasping Order (can be modified)
    
    # Extract volumes from volumes XMLs
    volume_after_first_grasp = volumes_xml_1_df.loc[volumes_xml_1_df['Index'] == index, 'Volume'].values
    volume_after_both_grasps = volumes_xml_2_df.loc[volumes_xml_2_df['Index'] == index, 'Volume'].values
    
    if len(volume_after_first_grasp) > 0:
        volume_after_first_grasp = volume_after_first_grasp[0]
    else:
        volume_after_first_grasp = 'Error'
    
    if len(volume_after_both_grasps) > 0:
        volume_after_both_grasps = volume_after_both_grasps[0]
    else:
        volume_after_both_grasps = 'Error'
    
    # Append processed data
    processed_data["Object 1 Volume"].append(obj1_volume)
    processed_data["Object 1 Grasping Length"].append(obj1_grasping_length)
    processed_data["Object 1 Shape (Cuboid, Sphere, Cylinder)"].append(obj1_shape_one_hot)
    processed_data["Object 2 Volume"].append(obj2_volume)
    processed_data["Object 2 Grasping Length"].append(obj2_grasping_length)
    processed_data["Object 2 Shape (Cuboid, Sphere, Cylinder)"].append(obj2_shape_one_hot)
  
    processed_data["Remaining Workspace Volume after first grasp"].append(volume_after_first_grasp)
    processed_data["Remaining Workspace Volume after both grasps"].append(volume_after_both_grasps)

# Convert processed data to DataFrame
final_df = pd.DataFrame(processed_data)

# Save the final DataFrame as a CSV file
final_csv_path = 'data/final_workspace_volumes.csv'
final_df.to_csv(final_csv_path, index=False)

final_csv_path
