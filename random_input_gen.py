import pandas as pd
import numpy as np

# Define the number of samples to generate
num_samples = 1000

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

# Function to generate shape one-hot encoding
def generate_shape():
    shape_type = np.random.choice(["Cuboid", "Sphere", "Cylinder"])
    if shape_type == "Cuboid":
        shape_encoding = [1, 0, 0]
        dimensions = generate_cube_dimensions()
    elif shape_type == "Sphere":
        shape_encoding = [0, 1, 0]
        dimensions = generate_sphere_dimensions()
    else:  # Cylinder
        shape_encoding = [0, 0, 1]
        dimensions = generate_cylinder_dimensions()
    return shape_encoding, dimensions

# Generate the dataset
data = {
    "Object 1 Shape": [],
    "Obj 1 Dimensions": [],
    "Object 2 Shape": [],
    "Obj 2 Dimensions": []
}

for _ in range(num_samples):
    obj1_shape, obj1_dimensions = generate_shape()
    obj2_shape, obj2_dimensions = generate_shape()
    data["Object 1 Shape"].append(obj1_shape)
    data["Obj 1 Dimensions"].append(obj1_dimensions)
    data["Object 2 Shape"].append(obj2_shape)
    data["Obj 2 Dimensions"].append(obj2_dimensions)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = 'random_object_shapes_rounded.csv'
print(f'CSV saved to {csv_path}')
df.to_csv(csv_path, index=False)

csv_path
