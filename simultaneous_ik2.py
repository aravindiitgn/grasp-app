import mujoco_py
import numpy as np

class OnlyPosIK:
    def __init__(self, xml_path):
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.jacp = np.zeros(3 * self.model.nv)  # Flattened Translation Jacobian
        self.jacr = np.zeros(3 * self.model.nv)  # Flattened Rotation Jacobian
        self.step_size = 0.5
        self.tol = 0.01
        self.alpha = 0.5
        self.init_q = np.zeros(self.sim.data.qpos.shape[0])

        # Define mapping from site names to joint indices
        self.site_to_joint_mapping = {
            "tip1": [0, 1, 2, 3],
            "tip2": [4, 5, 6, 7],  # Example joint indices for tip2
            "tip3": [8, 9, 10, 11],
            "tip4": [12, 13, 14, 15]  # Example joint indices for tip4
        }

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

                # Map site to its joint indices
                joint_indices = self.site_to_joint_mapping.get(site_name, [])
                grad_full = np.zeros_like(self.sim.data.qpos)
                grad_full[joint_indices] = grad[joint_indices]  # Restrict updates to specific joints
                
                total_grad += grad_full

            # Apply the total gradient to the allowed joints
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
        joint_indices = self.site_to_joint_mapping.get(site_name, [])
        return self.sim.data.qpos[joint_indices]

if __name__ == "__main__":
    xml_path = "/home/linux/Documents/mujoco200_linux/leap hand ws/leap ik3.xml"
    ik_solver = OnlyPosIK(xml_path)

    # Define target positions and site names for index finger and thumb
    goals = [np.array([-0.01, 0.01, 0.35]),
        np.array([-0.04, 0.065, 0.25]),  # Goal for index fingertip
        np.array([-0.075, 0.01, 0.3])  # Goal for thumb fingertip
    ]
    site_names = ["tip1","tip2", "tip4"]  # Sites for index finger and thumb

    # Perform simultaneous IK for both fingers
    ik_solver.calculate_simultaneous(goals, site_names)

    # Render the simulation to visualize
    while True:
        ik_solver.viewer.render()
