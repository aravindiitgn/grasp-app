import pandas as pd
import ast
import numpy as np

# File paths
shapes_csv_path = '../random_object_shapes_rounded.csv'
volumes_xml_1_path = 'xml_1/remaining_workspace_volumes.csv'
volumes_xml_2_path = 'xml_2/remaining_workspace_volumes_2.csv'

# Read CSV files
shapes_df = pd.read_csv(shapes_csv_path)

# Read volumes dataframes and handle missing values
volumes_xml_1_df = pd.read_csv(volumes_xml_1_path)
volumes_xml_2_df = pd.read_csv(volumes_xml_2_path)

# Extract index from sample_#.xml pattern
# Assuming 'sample' column is named as per example:
# sample,remaining_workspace_volume
# sample_0.xml,0.0031333734952219425
# ...
volumes_xml_1_df['Index'] = volumes_xml_1_df['sample'].str.extract(r'sample_(\d+)', expand=False).astype(int)
volumes_xml_2_df['Index'] = volumes_xml_2_df['sample'].str.extract(r'sample_(\d+)', expand=False).astype(int)

# Drop rows with NaN values in volume columns
volumes_xml_1_df.dropna(subset=['remaining_workspace_volume'], inplace=True)
volumes_xml_2_df.dropna(subset=['remaining_workspace_volume'], inplace=True)

# Rename columns for easier merging
volumes_xml_1_df.rename(columns={'remaining_workspace_volume': 'Volume_1'}, inplace=True)
volumes_xml_2_df.rename(columns={'remaining_workspace_volume': 'Volume_2'}, inplace=True)

# Merge volume data with shapes on Index
# It's possible that some Index values do not appear in either volume df after dropna
# So we will merge later and only process rows present in all dfs.

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

# Iterate over shapes_df
for index, row in shapes_df.iterrows():
    # Check if this index exists in both volume dfs
    vol_1_row = volumes_xml_1_df.loc[volumes_xml_1_df['Index'] == index]
    vol_2_row = volumes_xml_2_df.loc[volumes_xml_2_df['Index'] == index]

    # If either is empty, skip
    if vol_1_row.empty or vol_2_row.empty:
        continue

    volume_after_first_grasp = vol_1_row['Volume_1'].values[0]
    volume_after_both_grasps = vol_2_row['Volume_2'].values[0]

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

    # Append processed data
    processed_data["Object 1 Volume"].append(obj1_volume)
    processed_data["Object 1 Grasping Length"].append(obj1_grasping_length)
    processed_data["Object 1 Shape (Cuboid, Sphere, Cylinder)"].append(obj1_shape)
    processed_data["Object 2 Volume"].append(obj2_volume)
    processed_data["Object 2 Grasping Length"].append(obj2_grasping_length)
    processed_data["Object 2 Shape (Cuboid, Sphere, Cylinder)"].append(obj2_shape)
    processed_data["Remaining Workspace Volume after first grasp"].append(volume_after_first_grasp)
    processed_data["Remaining Workspace Volume after both grasps"].append(volume_after_both_grasps)

# Convert processed data to DataFrame
final_df = pd.DataFrame(processed_data)

# Save the final DataFrame as a CSV file
final_csv_path = 'final_workspace_volumes.csv'
final_df.to_csv(final_csv_path, index=False)


