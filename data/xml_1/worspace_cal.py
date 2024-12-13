import mujoco_py
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
class OnlyPosIK:
    def __init__(self, xml_path, viewer=None):
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = viewer if viewer else mujoco_py.MjViewer(self.sim)
        self.jacp = np.zeros(3 * self.model.nv)  # Flattened Translation Jacobian
        self.jacr = np.zeros(3 * self.model.nv)  # Flattened Rotation Jacobian
        self.step_size = 0.5
        self.tol = 0.01
        self.alpha = 0.5
        self.init_q = np.zeros(self.sim.data.qpos.shape[0])
    def calculate_simultaneous(self, goals, site_names):
        """Perform IK for multiple sites simultaneously."""
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

            # Apply the total gradient to all joints
            self.sim.data.qpos[:] += self.step_size * total_grad
            self.sim.forward()
            # Update errors
            errors = [np.subtract(goal, self.sim.data.site_xpos[site_id]) for goal, site_id in zip(goals, site_ids)]
            iteration += 1

        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations reached. The solution may not have converged.")
        
        for site_name, site_id in zip(site_names, site_ids):
            print(f"\nFinal joint positions for site '{site_name}':")
            joint_angles = self.get_joint_angles_for_site(site_name)
            print(f"Joint angles: {joint_angles}")

        return self.sim.data.qpos.copy()

    def get_joint_angles_for_site(self, site_name):
        """Retrieve joint angles for the site."""
        # Map site names to joint indices
        site_to_joint_mapping = {
            "tip1": [0, 1, 2, 3],  # Example joint indices for tip1
                                      # Example joint indices for tip2 (tip3 in your XML)
            "tip2": [4, 5, 6, 7],
            "tip3": [8, 9, 10, 11],
            "tip4": [12, 13, 14, 15]    # Example joint indices for third (middle) finger tip
        }
        joint_indices = site_to_joint_mapping.get(site_name, [])
        return self.sim.data.qpos[joint_indices]

    def find_workspace_in_angles(self, fixed_goals, fixed_site_names, index_finger_joint_ranges,thumb_finger_joint_range):
        """Find the complete workspace of the third finger in joint space."""
        # Fix the index and thumb fingers first
        qpos_init = self.calculate_simultaneous(fixed_goals, fixed_site_names)

        # Generate possible joint angles for the third finger
        workspacei_positions = []
        workspacet_positions = []
        joint_ranges1 = index_finger_joint_ranges  # list of (min, max) tuples for each joint of third finger
        joint_ranges2 = thumb_finger_joint_range
        # Iterate over all possible joint angles within the specified range for each joint
        for joint1i_angle in np.linspace(joint_ranges1[0][0], joint_ranges1[0][1],4):
            for joint2i_angle in np.linspace(joint_ranges1[1][0], joint_ranges1[1][1], 4):
                for joint3i_angle in np.linspace(joint_ranges1[2][0], joint_ranges1[2][1], 4):
                    for joint4i_angle in np.linspace(joint_ranges1[3][0], joint_ranges1[3][1], 4):
                        for joint1t_angle in np.linspace(joint_ranges2[0][0], joint_ranges2[0][1], 3):
                            for joint2t_angle in np.linspace(joint_ranges2[1][0], joint_ranges2[1][1], 3):
                                for joint3t_angle in np.linspace(joint_ranges2[2][0], joint_ranges2[2][1], 3):
                                    for joint4t_angle in np.linspace(joint_ranges2[3][0], joint_ranges2[3][1], 3):
                                        # Set the joint angles for the third finger
                                        index_finger_angles = np.array([joint1i_angle, joint2i_angle, joint3i_angle, joint4i_angle])
                                        thumb_finger_angles = np.array([joint1t_angle, joint2t_angle, joint3t_angle, joint4t_angle])
                                        # Update the third finger's joint angles
                                        self.sim.data.qpos[0:4] = index_finger_angles
                                        self.sim.data.qpos[4:12] = qpos_init[4:12]
                                        self.sim.data.qpos[12:16] = thumb_finger_angles

                                        # Assuming the indices for third finger are 4 to 7
                                        self.sim.step()
                                        

                                        # Get the position of the third finger's fingertip
                                        fingertip2_position = self.sim.data.site_xpos[self.model.site_name2id("tip1")].copy()
                                        fingertip3_position = self.sim.data.site_xpos[self.model.site_name2id("tip4")].copy()
                                        workspacei_positions.append(fingertip2_position)
                                        workspacet_positions.append(fingertip3_position)
                                        self.viewer.render()
        workspacei = np.array(workspacei_positions)
        workspacet = np.array(workspacet_positions)

        np.save("WorkspaceI", workspacei)
        np.save("WorkspaceT", workspacet)


        return  workspacei, workspacet

    def visualize_convex_hull(self, workspacei_positions, workspacet_positions):
        if workspacei_positions is None or workspacet_positions is None or len(workspacei_positions) == 0 or len(workspacet_positions) == 0:
            raise ValueError("Workspace positions are empty. Ensure that the input is not empty.")
        
        # Process input workspace positions
        workspacei = np.asarray(workspacei_positions)
        workspacet = np.asarray(workspacet_positions)
        workspacei_points = np.unique(workspacei, axis=0)
        workspacet_points = np.unique(workspacet, axis=0)

        if len(workspacei_points) < 4 or len(workspacet_points) < 4:
            print("Not enough unique points for ConvexHull in one or both workspaces.")
            return

        # Add noise to avoid coplanarity issues
        workspacei_points += np.random.normal(scale=1e-5, size=workspacei_points.shape)
        workspacet_points += np.random.normal(scale=1e-5, size=workspacet_points.shape)
        combined_points = np.vstack((workspacei_points, workspacet_points))

        try:
            hull = ConvexHull(combined_points)
            workspace_volume = hull.volume
            print(f"Combined Workspace Volume: {workspace_volume} cubic units")
        except Exception as e:
            print("ConvexHull calculation failed:", e)
            return

        # Determine the object geometry
        object_geom_id = self.sim.model.geom_name2id('box_geom')  # Replace with actual object name
        geom_type = self.sim.model.geom_type[object_geom_id]  # Geometry type
        geom_size = self.sim.model.geom_size[object_geom_id]  # Size parameters
        object_position = self.sim.data.body_xpos[object_geom_id]
        geom_name = self.sim.model.geom_id2name(object_geom_id)

        print(f"Geometry Name: {geom_name}, Geometry Type: {geom_type}, Size: {geom_size}")

        # Generate points for different object types
        if geom_type == 2:  # Sphere
            radius = geom_size[0]
            num_samples = 1000
            phi = np.random.uniform(0, 2 * np.pi, num_samples)
            costheta = np.random.uniform(-1, 1, num_samples)
            u = np.random.uniform(0, 1, num_samples)
            theta = np.arccos(costheta)
            r = radius * u ** (1/3)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            object_points = np.vstack((x, y, z)).T + object_position
        elif geom_type == 6:  # Box
            half_extents = geom_size
            offsets = np.array([
                [-1, -1, -1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1,  1,  1],
                [ 1, -1, -1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [ 1,  1,  1]
            ]) * half_extents
            object_points = offsets + object_position
        elif geom_type == 5:  # Cylinder
            radius = geom_size[0]
            height = geom_size[1]
            num_samples = 5000
            angles = np.random.uniform(0, 2 * np.pi, num_samples)
            heights = np.random.uniform(-height / 2, height / 2, num_samples)
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            z = heights
            object_points = np.vstack((x, y, z)).T + object_position
            object_points += np.random.normal(scale=1e-5, size=object_points.shape)
        else:
            raise ValueError("Unsupported geometry type. Extend this code to handle more shapes.")

        # Create the Delaunay triangulation for the object
        try:
            object_delaunay = Delaunay(object_points, qhull_options='QJ')
        except scipy.spatial.qhull.QhullError as e:
            print("Delaunay triangulation failed:", e)
            object_delaunay = None


        # Create the convex hull for the object
        try:
            object_hull = ConvexHull(object_points, qhull_options='QJ')
            object_volume = object_hull.volume
            print(f"Object Convex Hull Volume: {object_volume} cubic units")
        except Exception as e:
            print("ConvexHull calculation failed for the object:", e)
            return

        # Compute intersection
        workspace_delaunay = Delaunay(combined_points)
        object_delaunay = Delaunay(object_points)

        min_bounds = np.min(combined_points, axis=0)
        max_bounds = np.max(combined_points, axis=0)
        grid_x, grid_y, grid_z = np.mgrid[
            min_bounds[0]:max_bounds[0]:50j,
            min_bounds[1]:max_bounds[1]:50j,
            min_bounds[2]:max_bounds[2]:50j
        ]
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

        inside_workspace = workspace_delaunay.find_simplex(grid_points) >= 0
        inside_object = object_delaunay.find_simplex(grid_points) >= 0
        intersection_points = grid_points[inside_workspace & inside_object]

        if len(intersection_points) > 3:
            intersection_hull = ConvexHull(intersection_points)
            intersection_volume = intersection_hull.volume
        else:
            intersection_volume = 0

        print(f"Intersection Volume: {intersection_volume} cubic units")
        remaining_workspace_volume = workspace_volume - intersection_volume
        print(f"Remaining Workspace Volume: {remaining_workspace_volume} cubic units")
        results.append({
                'sample': f'sample_{i}.xml',
                'remaining_workspace_volume': remaining_workspace_volume
            })
        # Visualization (unchanged, update for specific needs)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(workspacei_points[:, 0], workspacei_points[:, 1], workspacei_points[:, 2], c='b', marker='o', alpha=0.5, label="Index Finger")
        ax.scatter(workspacet_points[:, 0], workspacet_points[:, 1], workspacet_points[:, 2], c='r', marker='o', alpha=0.5, label="Thumb")
        for simplex in hull.simplices:
            triangle = combined_points[simplex]
            poly = Poly3DCollection([triangle], alpha=0.5, edgecolor='k')
            poly.set_facecolor('cyan')
            ax.add_collection3d(poly)
        ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c='g', label="Object Points")
        ax.set_title("Convex Hull of Workspaces and Object")
        ax.legend()
        plt.show()



if __name__ == "__main__":
    results = []
    viewer = None  # Initialize viewer variable
    for i in range(1000): 
        try:
            print(f"Processing iteration {i}...")
            xml_path = f"sample_{i}.xml"
            ik_solver = OnlyPosIK(xml_path, viewer=viewer)
            viewer = ik_solver.viewer  # Reuse the same viewer instance
            
            fixed_goals = [np.array([-0.055, -0.020, 0.12]), np.array([-0.055, -0.065, 0.12])]
            fixed_site_names = ["tip2", "tip3"]

            index_finger_joint_ranges = [
                (-0.314, 2.23), (-1.047, 0.12), (-0.506, 1.885), (-0.366, 2.042)
            ]
            thumb_finger_joint_ranges = [
                (0.349, 2.094), (-0.47, 2.443), (-1.2, 1.9), (-1.34, 1.88)
            ]

            # Get the workspace positions for both fingers
            workspacei_positions, workspacet_positions = ik_solver.find_workspace_in_angles(
                fixed_goals, fixed_site_names, index_finger_joint_ranges, thumb_finger_joint_ranges
            )
            
            # Pass both workspaces to the visualization method
            ik_solver.visualize_convex_hull(workspacei_positions, workspacet_positions)

        except Exception as e:
            print(f"Error processing sample_{i}.xml at iteration {i}: {e}")
            results.append({
                'sample': f'sample_{i}.xml',
                'remaining_workspace_volume': None
            })
            continue

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('remaining_workspace_volumes.csv', index=False)
    print("Results saved to remaining_workspace_volumes.csv")
