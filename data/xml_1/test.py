import pandas as pd
import numpy as np
import os
import mujoco_py
from scipy.spatial import ConvexHull, Delaunay


class OnlyPosIK:
    def __init__(self, xml_path):
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.jacp = np.zeros(3 * self.model.nv)  # Flattened Translation Jacobian
        self.jacr = np.zeros(3 * self.model.nv)  # Flattened Rotation Jacobian
        self.step_size = 0.5
        self.tol = 0.01
        self.alpha = 0.5
        self.init_q = np.zeros(self.sim.data.qpos.shape[0])

    def calculate_simultaneous(self, goals, site_names):
        site_ids = [self.model.site_name2id(name) for name in site_names]
        self.sim.data.qpos[:] = self.init_q
        self.sim.forward()

        max_iterations = 1000
        iteration = 0
        errors = [np.subtract(goal, self.sim.data.site_xpos[site_id]) for goal, site_id in zip(goals, site_ids)]

        while all(np.linalg.norm(err) >= self.tol for err in errors) and iteration < max_iterations:
            total_grad = np.zeros_like(self.sim.data.qpos)
            for goal, site_name, site_id, error in zip(goals, site_names, site_ids, errors):
                mujoco_py.functions.mj_jacSite(self.model, self.sim.data, self.jacp, self.jacr, site_id)
                jacp_matrix = self.jacp.reshape(3, self.model.nv)
                grad = self.alpha * jacp_matrix.T @ error
                grad_full = np.zeros_like(self.sim.data.qpos)
                grad_full[: len(grad)] = grad
                total_grad += grad_full

            self.sim.data.qpos[:] += self.step_size * total_grad
            self.sim.forward()
            errors = [np.subtract(goal, self.sim.data.site_xpos[site_id]) for goal, site_id in zip(goals, site_ids)]
            iteration += 1

        return self.sim.data.qpos.copy()

    def find_workspace_in_angles(self, fixed_goals, fixed_site_names, index_finger_joint_ranges, thumb_finger_joint_range):
        qpos_init = self.calculate_simultaneous(fixed_goals, fixed_site_names)
        workspacei_positions = []
        workspacet_positions = []

        for joint1i_angle in np.linspace(index_finger_joint_ranges[0][0], index_finger_joint_ranges[0][1], 4):
            for joint2i_angle in np.linspace(index_finger_joint_ranges[1][0], index_finger_joint_ranges[1][1], 4):
                for joint3i_angle in np.linspace(index_finger_joint_ranges[2][0], index_finger_joint_ranges[2][1], 4):
                    for joint4i_angle in np.linspace(index_finger_joint_ranges[3][0], index_finger_joint_ranges[3][1], 4):
                        for joint1t_angle in np.linspace(thumb_finger_joint_range[0][0], thumb_finger_joint_range[0][1], 3):
                            for joint2t_angle in np.linspace(thumb_finger_joint_range[1][0], thumb_finger_joint_range[1][1], 3):
                                for joint3t_angle in np.linspace(thumb_finger_joint_range[2][0], thumb_finger_joint_range[2][1], 3):
                                    for joint4t_angle in np.linspace(thumb_finger_joint_range[3][0], thumb_finger_joint_range[3][1], 3):
                                        index_finger_angles = np.array([joint1i_angle, joint2i_angle, joint3i_angle, joint4i_angle])
                                        thumb_finger_angles = np.array([joint1t_angle, joint2t_angle, joint3t_angle, joint4t_angle])
                                        self.sim.data.qpos[0:4] = index_finger_angles
                                        self.sim.data.qpos[4:12] = qpos_init[4:12]
                                        self.sim.data.qpos[12:16] = thumb_finger_angles
                                        self.sim.step()

                                        fingertip2_position = self.sim.data.site_xpos[self.model.site_name2id("tip1")].copy()
                                        fingertip3_position = self.sim.data.site_xpos[self.model.site_name2id("tip4")].copy()
                                        workspacei_positions.append(fingertip2_position)
                                        workspacet_positions.append(fingertip3_position)
        return np.array(workspacei_positions), np.array(workspacet_positions)


if __name__ == "__main__":
    results = []  # To store the results for each sample
    for i in range(1000):  # From sample_0.xml to sample_999.xml
        try:
            xml_path = f"sample_{i}.xml"
            if not os.path.isfile(xml_path):
                print(f"File {xml_path} does not exist. Skipping...")
                continue

            ik_solver = OnlyPosIK(xml_path)
            fixed_goals = [np.array([-0.055, -0.020, 0.12]), np.array([-0.055, -0.065, 0.12])]
            fixed_site_names = ["tip2", "tip3"]

            index_finger_joint_ranges = [(-0.314, 2.23), (-1.047, 0.12), (-0.506, 1.885), (-0.366, 2.042)]
            thumb_finger_joint_ranges = [(0.349, 2.094), (-0.47, 2.443), (-1.2, 1.9), (-1.34, 1.88)]

            workspacei_positions, workspacet_positions = ik_solver.find_workspace_in_angles(
                fixed_goals, fixed_site_names, index_finger_joint_ranges, thumb_finger_joint_ranges
            )

            combined_points = np.vstack((workspacei_positions, workspacet_positions))
            try:
                hull = ConvexHull(combined_points)
                workspace_volume = hull.volume
            except Exception as e:
                print(f"ConvexHull calculation failed: {e}")
                continue

            try:
                object_hull = ConvexHull(combined_points)
                object_volume = object_hull.volume
            except Exception as e:
                print(f"ConvexHull calculation failed for the object: {e}")
                continue

            try:
                intersection_hull = ConvexHull(combined_points)
                intersection_volume = intersection_hull.volume
            except Exception as e:
                intersection_volume = 0

            remaining_workspace_volume = workspace_volume - intersection_volume

            results.append({
                'sample': f'sample_{i}.xml',
                'remaining_workspace_volume': remaining_workspace_volume
            })

            print(f"Processed sample_{i}.xml: Remaining Workspace Volume = {remaining_workspace_volume}")

        except Exception as e:
            print(f"Error processing sample_{i}.xml: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('remaining_workspace_volumes.csv', index=False)
    print("Results saved to remaining_workspace_volumes.csv")
