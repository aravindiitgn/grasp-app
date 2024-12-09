# grasp-plan-multiobject


# **Workspace Volume Data Processing Script - dataset_get.py**


This script processes three CSV files to generate a **final CSV file** containing object details, grasping lengths, and workspace volumes after one and two grasps.

---

## **Input Files**
1. **`random_object_shapes_rounded.csv`**: Contains shape and dimensions for two objects.
2. **`data/workspace_volumes_xml_1.csv`**: Contains remaining workspace volume after **first grasp**.
3. **`data/workspace_volumes_xml_2.csv`**: Contains remaining workspace volume after **both grasps**.

---

## **Output File**
- **`data/final_workspace_volumes.csv`**
  - **Structure**:
    ```
    Object 1 Volume, Object 1 Grasping Length, Object 1 Shape (One-Hot), 
    Object 2 Volume, Object 2 Grasping Length, Object 2 Shape (One-Hot), 
    Remaining Workspace Volume after first grasp, Remaining Workspace Volume after both grasps
    ```


---

## **Key Columns in Final Dataset CSV**
| **Column**                                 | **Description**                          |
|--------------------------------------------|------------------------------------------|
| **Object 1 Volume**                        | Volume of Object 1 (based on dimensions) |
| **Object 1 Grasping Length**               | 2x Y-dimension (for Cuboid/Cylinder)    |
| **Object 1 Shape (Cuboid, Sphere, Cylinder)** | One-hot encoding of the shape of Object 1 (e.g., [1, 0, 0] for Cuboid) |
| **Object 2 Volume**                        | Volume of Object 2                      |
| **Object 2 Grasping Length**               | 2x Y-dimension (for Cuboid/Cylinder)    |
| **Object 2 Shape (Cuboid, Sphere, Cylinder)** | One-hot encoding of the shape of Object 2 (e.g., [0, 1, 0] for Sphere) |
| **Remaining Workspace Volume after first grasp** | Extracted from **workspace_volumes_xml_1.csv** |
| **Remaining Workspace Volume after both grasps** | Extracted from **workspace_volumes_xml_2.csv** |

