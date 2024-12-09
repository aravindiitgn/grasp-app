import xml.etree.ElementTree as ET

def modify_box_geom_size(xml_file, new_size):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the box_geom element within the body named 'cube'
    for body in root.findall(".//body[@name='cube']"):
        for geom in body.findall(".//geom[@name='box_geom']"):
            # Update the size attribute
            geom.set('size', new_size)

    # Save the modified XML to a new file
    modified_file = "modified_" + xml_file
    tree.write(modified_file, encoding="UTF-8", xml_declaration=True)
    print(f"Modified XML saved as: {modified_file}")

# Input XML file
input_xml = "input_xml.xml"

# New size for box_geom
new_size_value = "0.001 0.002 0.003"

# Modify the box_geom size
modify_box_geom_size(input_xml, new_size_value)
