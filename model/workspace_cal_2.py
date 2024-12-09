import os
import pandas as pd
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy.spatial import ConvexHull


class OnlyPosIK:
    def __init__(self, xml_path):
        self.model = load_model_from_path(xml_path)
        self.sim = MjSim(self.model)
        self.jacp = np.zeros(3 * self.model.nv)  # Flattened Translation Jacobian
        self.jacr = np.zeros(3 * self.model.nv)  # Flattened Rotation Jacobian

    def find_workspace_in_angles(self, fixed_goals, fixed_site_names, pinky_finger_joint_ranges):
        workspacep_positions = []
        
        joint_ranges = pinky_finger_joint_ranges

        for joint1p_angle in np.linspace(joint_ranges[0][0], joint_ranges[0][1], 20):
            for joint2p_angle in np.linspace(joint_ranges[1][0], joint_ranges[1][1], 20):
                for joint3p_angle in np.linspace(joint_ranges[2][0], joint_ranges[2][1], 20):
                    for joint4p_angle in np.linspace(joint_ranges[3][0], joint_ranges[3][1], 20):
                        pinky_finger_angles = np.array([joint1p_angle, joint2p_angle, joint3p_angle, joint4p_angle])
                        
                        self.sim.data.qpos[8:12] = pinky_finger_angles
                        
                        self.sim.step()
                        
                        fingertip_position = self.sim.data.site_xpos[self.model.site_name2id("tip3")].copy()
                        workspacep_positions.append(fingertip_position)

        workspacep = np.array(workspacep_positions)
        return workspacep

    def calculate_workspace_volume(self, workspacep_positions):
        if len(workspacep_positions) < 4:
            print("Not enough points to compute Convex Hull.")
            return None

        try:
            hull = ConvexHull(workspacep_positions)
            return hull.volume
        except Exception as e:
            print(f"Error calculating ConvexHull: {e}")
            return None


# Directory containing the XML files
xml_directory = '../data/xml_2/'

# Initialize the CSV file to store the results
results = []

# Loop through all XML files and compute the workspace volume
for index in range(1000):
    xml_path = os.path.join(xml_directory, f'sample{index}.xml')
    
    if os.path.exists(xml_path):
        print(f"Processing {xml_path}...")

        try:
            ik_solver = OnlyPosIK(xml_path)
            
            fixed_goals = [np.array([-0.01, 0.01, 0.35]), np.array([-0.055, 0.055, 0.20]), np.array([-0.075, 0.01, 0.3])]
            fixed_site_names = ["tip1", "tip2", "tip4"]

            pinky_finger_joint_ranges = [(-0.314, 2.23), (0, 1.047), (-0.506, 1.885), (-0.366, 2.042)]

            workspacep_positions = ik_solver.find_workspace_in_angles(
                fixed_goals, fixed_site_names, pinky_finger_joint_ranges
            )

            volume = ik_solver.calculate_workspace_volume(workspacep_positions)
            results.append({'Index': index, 'Volume': volume if volume else 'Error'})

        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            results.append({'Index': index, 'Volume': 'Error'})

    else:
        print(f"File {xml_path} does not exist.")
        results.append({'Index': index, 'Volume': 'File Not Found'})

# Save results to CSV
df_results = pd.DataFrame(results)
csv_path = '../data/workspace_volumes_xml_2.csv'
df_results.to_csv(csv_path, index=False)

print(f"CSV file saved at: {csv_path}")
