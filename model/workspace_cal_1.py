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

    def find_workspace_in_angles(self, fixed_goals, fixed_site_names, index_finger_joint_ranges, thumb_finger_joint_range):
        workspacei_positions = []
        workspacet_positions = []

        joint_ranges1 = index_finger_joint_ranges
        joint_ranges2 = thumb_finger_joint_range

        for joint1i_angle in np.linspace(joint_ranges1[0][0], joint_ranges1[0][1], 5):
            for joint2i_angle in np.linspace(joint_ranges1[1][0], joint_ranges1[1][1], 5):
                for joint3i_angle in np.linspace(joint_ranges1[2][0], joint_ranges1[2][1], 5):
                    for joint4i_angle in np.linspace(joint_ranges1[3][0], joint_ranges1[3][1], 5):
                        for joint1t_angle in np.linspace(joint_ranges2[0][0], joint_ranges2[0][1], 5):
                            for joint2t_angle in np.linspace(joint_ranges2[1][0], joint_ranges2[1][1], 5):
                                for joint3t_angle in np.linspace(joint_ranges2[2][0], joint_ranges2[2][1], 5):
                                    for joint4t_angle in np.linspace(joint_ranges2[3][0], joint_ranges2[3][1], 5):
                                        index_finger_angles = np.array([joint1i_angle, joint2i_angle, joint3i_angle, joint4i_angle])
                                        thumb_finger_angles = np.array([joint1t_angle, joint2t_angle, joint3t_angle, joint4t_angle])

                                        self.sim.data.qpos[0:4] = index_finger_angles
                                        self.sim.data.qpos[12:16] = thumb_finger_angles

                                        self.sim.step()
                                        
                                        fingertip2_position = self.sim.data.site_xpos[self.model.site_name2id("tip1")].copy()
                                        fingertip3_position = self.sim.data.site_xpos[self.model.site_name2id("tip4")].copy()
                                        workspacei_positions.append(fingertip2_position)
                                        workspacet_positions.append(fingertip3_position)

        workspacei = np.array(workspacei_positions)
        workspacet = np.array(workspacet_positions)

        return workspacei, workspacet

    def calculate_workspace_volume(self, workspacei_positions, workspacet_positions):
        combined_points = np.vstack((workspacei_positions, workspacet_positions))
        
        if len(combined_points) < 4:
            print("Not enough points to compute Convex Hull.")
            return None

        try:
            hull = ConvexHull(combined_points)
            return hull.volume
        except Exception as e:
            print(f"Error calculating ConvexHull: {e}")
            return None


# Directory containing the XML files
xml_directory = '../data/xml_1/'

# Initialize the CSV file to store the results
results = []

# Loop through all XML files and compute the workspace volume
for index in range(1000):
    xml_path = os.path.join(xml_directory, f'sample{index}.xml')
    
    if os.path.exists(xml_path):
        print(f"Processing {xml_path}...")

        try:
            ik_solver = OnlyPosIK(xml_path)
            
            fixed_goals = [np.array([-0.055, -0.020, 0.12]), np.array([-0.055, -0.065, 0.12])]
            fixed_site_names = ["tip2", "tip3"]

            index_finger_joint_ranges = [(-0.314, 2.23), (-1.047, 1.047), (-0.506, 1.885), (-0.366, 2.042)]
            thumb_finger_joint_ranges = [(0.349, 2.094), (-0.47, 2.443), (-1.2,1.9), (-1.34, 1.88)]

            workspacei_positions, workspacet_positions = ik_solver.find_workspace_in_angles(
                fixed_goals, fixed_site_names, index_finger_joint_ranges, thumb_finger_joint_ranges
            )

            volume = ik_solver.calculate_workspace_volume(workspacei_positions, workspacet_positions)
            results.append({'Index': index, 'Volume': volume if volume else 'Error'})

        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            results.append({'Index': index, 'Volume': 'Error'})

    else:
        print(f"File {xml_path} does not exist.")
        results.append({'Index': index, 'Volume': 'File Not Found'})

# Save results to CSV
df_results = pd.DataFrame(results)
csv_path = '../data/workspace_volumes_xml_1.csv'
df_results.to_csv(csv_path, index=False)

print(f"CSV file saved at: {csv_path}")
