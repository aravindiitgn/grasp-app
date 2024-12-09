import pandas as pd
import xml.etree.ElementTree as ET

# Load the CSV file
csv_path = 'random_object_shapes_rounded.csv'  # Path to the CSV file
df = pd.read_csv(csv_path)

# Loop through each row of the CSV
for index, row in df.iterrows():
    # Parse the existing XML file
    tree = ET.parse('model/second_ws.xml')  # Path to the original XML file
    root = tree.getroot()

    # Extract Object 1 information (to be placed in <body name="cube">)
    obj1_shape = eval(row["Object 1 Shape"])  # Convert string back to list
    obj1_dimensions = eval(row["Obj 1 Dimensions"])  # Convert string back to list

    # Determine object 1 type and size
    if obj1_shape == [1, 0, 0]:  # Cuboid
        obj1_type = "box"
        obj1_size = f'{obj1_dimensions[0]} {obj1_dimensions[1]} {obj1_dimensions[2]}'
    elif obj1_shape == [0, 1, 0]:  # Sphere
        obj1_type = "sphere"
        obj1_size = f'{obj1_dimensions[0]}'
    elif obj1_shape == [0, 0, 1]:  # Cylinder
        obj1_type = "cylinder"
        obj1_size = f'{obj1_dimensions[0]} {obj1_dimensions[1]}'

    # Update <body name="cube"> with Object 1 data
    for body in root.findall(".//body[@name='cube']"):
        geom = body.find(".//geom[@name='box_geom']")
        if geom is not None:
            geom.set('type', obj1_type)
            geom.set('size', obj1_size)

    # Extract Object 2 information (to be placed in <body name="cube1">)
    obj2_shape = eval(row["Object 2 Shape"])  # Convert string back to list
    obj2_dimensions = eval(row["Obj 2 Dimensions"])  # Convert string back to list

    # Determine object 2 type and size
    if obj2_shape == [1, 0, 0]:  # Cuboid
        obj2_type = "box"
        obj2_size = f'{obj2_dimensions[0]} {obj2_dimensions[1]} {obj2_dimensions[2]}'
    elif obj2_shape == [0, 1, 0]:  # Sphere
        obj2_type = "sphere"
        obj2_size = f'{obj2_dimensions[0]}'
    elif obj2_shape == [0, 0, 1]:  # Cylinder
        obj2_type = "cylinder"
        obj2_size = f'{obj2_dimensions[0]} {obj2_dimensions[1]}'

    # Update <body name="cube1"> with Object 2 data
    for body in root.findall(".//body[@name='cube1']"):
        geom = body.find(".//geom[@name='box_geom']")
        if geom is not None:
            geom.set('type', obj2_type)
            geom.set('size', obj2_size)

    # Save the modified XML file with a new name for each row
    output_file_path = f'data/xml_2/sample_{index}.xml'
    tree.write(output_file_path)

    print(f"Saved modified XML file for row {index} at {output_file_path}")
