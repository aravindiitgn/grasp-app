import mujoco_py
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
class OnlyPosIK:
    def __init__(self, xml_path,viewer=None):
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

    def find_workspace_in_angles(self, fixed_goals, fixed_site_names, pinky_finger_joint_ranges):
        """Find the complete workspace of the third finger in joint space."""
        # Fix the index and thumb fingers first
        qpos_init = self.calculate_simultaneous(fixed_goals, fixed_site_names)

        # Generate possible joint angles for the third finger
        workspacep_positions = []
        
        joint_ranges = pinky_finger_joint_ranges  # list of (min, max) tuples for each joint of third finger
        
        # Iterate over all possible joint angles within the specified range for each joint
        for joint1p_angle in np.linspace(joint_ranges[0][0], joint_ranges[0][1],15):
            for joint2p_angle in np.linspace(joint_ranges[1][0], joint_ranges[1][1], 15):
                for joint3p_angle in np.linspace(joint_ranges[2][0], joint_ranges[2][1], 15):
                    for joint4p_angle in np.linspace(joint_ranges[3][0], joint_ranges[3][1], 15):
                       
                        # Set the joint angles for the third finger
                        pinky_finger_angles = np.array([joint1p_angle, joint2p_angle, joint3p_angle, joint4p_angle])
                        
                        # Update the third finger's joint angles
                        self.sim.data.qpos[8:12] = pinky_finger_angles
                        self.sim.data.qpos[0:8] = qpos_init[0:8]
                        self.sim.data.qpos[12:16] = qpos_init[12:16]
                        

                        # Assuming the indices for third finger are 4 to 7
                        self.sim.step()
                        

                        # Get the position of the third finger's fingertip
                        fingertip_position = self.sim.data.site_xpos[self.model.site_name2id("tip3")].copy()
                        
                        workspacep_positions.append(fingertip_position)
                        
                        self.viewer.render()
        workspacep = np.array(workspacep_positions)
        

        np.save("WorkspaceI", workspacep)
        


        return  workspacep

    def visualize_convex_hull(self, workspacep_positions):
        if workspacep_positions is None or len(workspacep_positions) == 0:
            raise ValueError("Workspace positions are empty. Ensure that the input is not empty.")
        
        # Ensure the arrays are 2D for np.unique to work correctly
        workspacep = np.asarray(workspacep_positions)
        if workspacep.ndim == 1:
            workspacep = workspacep.reshape(-1, 1)
        
        
        # Extract unique points for each workspace
        workspacep_points = np.unique(workspacep, axis=0)
        

        # Check for sufficient points for ConvexHull
        if len(workspacep_points) < 4:
            print("Not enough unique points for ConvexHull. At least 4 points are required for the index finger workspace.")
            return
        
        # Add slight noise to avoid coplanarity issues
        workspacep_points += np.random.normal(scale=1e-5, size=workspacep_points.shape)
        
        
        # Try constructing the convex hull
        try:
            hull = ConvexHull(workspacep_points)
            workspace_volume = hull.volume
            print(f"Combined Workspace Volume: {workspace_volume} cubic units")
        except Exception as e:
            print("ConvexHull calculation failed:", e)
            workspace_volume = None
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

        # Sample points inside workspace convex hull
        workspace_delaunay = Delaunay(workspacep_points)
        object_delaunay = Delaunay(object_points)

        # Generate a dense grid within workspace bounds
        min_bounds = np.min(workspacep_points, axis=0)
        max_bounds = np.max(workspacep_points, axis=0)
        grid_x, grid_y, grid_z = np.mgrid[
            min_bounds[0]:max_bounds[0]:50j,
            min_bounds[1]:max_bounds[1]:50j,
            min_bounds[2]:max_bounds[2]:50j
        ]
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        
        # Find points inside both hulls (intersection points)
        inside_workspace = workspace_delaunay.find_simplex(grid_points) >= 0
        inside_object = object_delaunay.find_simplex(grid_points) >= 0
        intersection_points = grid_points[inside_workspace & inside_object]

        # Compute intersection convex hull and its volume
        if len(intersection_points) > 3:  # Minimum points to form a convex hull
            intersection_hull = ConvexHull(intersection_points)
            intersection_volume = intersection_hull.volume
        else:
            intersection_volume = 0
        print(f"Intersection Volume: {intersection_volume} cubic units")

        # Calculate remaining workspace volume
        remaining_workspace_volume = workspace_volume - intersection_volume
        print(f"Remaining Workspace Volume: {remaining_workspace_volume} cubic units")
        results.append({
                'sample': f'sample_{i}.xml',
                'remaining_workspace_volume': remaining_workspace_volume
            })



        # Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(workspacep_points[:, 0], workspacep_points[:, 1], workspacep_points[:, 2], c='b', marker='o', alpha=0.2, label="pinky Finger")
        
        
        for simplex in hull.simplices:
            triangle = workspacep_points[simplex]
            poly = Poly3DCollection([triangle], alpha=0.5, edgecolor='k')
            poly.set_facecolor('cyan')
            ax.add_collection3d(poly)

        ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c='b', marker='^', label="Object Points")
        for simplex in object_hull.simplices:
            triangle = object_points[simplex]
            poly3d = [[tuple(triangle[0]), tuple(triangle[1]), tuple(triangle[2])]]
            ax.add_collection3d(Poly3DCollection(poly3d, color='green', alpha=0.5))

        # Plot intersection points
        if len(intersection_points) > 0:
            ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], c='purple', label="Intersection Points")    

        ax.set_title("Convex Hull of Finger Workspaces")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
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
            
            fixed_goals = [np.array([-0.01, 0.01, 0.35]), np.array([-0.055, 0.055, 0.20]), np.array([-0.075, 0.01, 0.3])]
            fixed_site_names = ["tip1", "tip2", "tip4"]

            pinky_finger_joint_ranges = [
                (-0.314, 2.23), (0, 1.047), (-0.506, 1.885), (-0.366, 2.042)
            ]
                    # Get the workspace positions for both fingers
            workspacep_positions = ik_solver.find_workspace_in_angles(
                fixed_goals, fixed_site_names, pinky_finger_joint_ranges
            )
            
            # Pass both workspaces to the visualization method
            ik_solver.visualize_convex_hull(workspacep_positions)


        except Exception as e:
            print(f"Error processing sample_{i}.xml at iteration {i}: {e}")
            results.append({
                'sample': f'sample_{i}.xml',
                'remaining_workspace_volume': None
            })
            continue

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('remaining_workspace_volumes_2.csv', index=False)
    print("Results saved to remaining_workspace_volumes.csv")


in this code i want to select the index range of xml files which it should process and it shouldn't overwrite the the csv file, it should add to it