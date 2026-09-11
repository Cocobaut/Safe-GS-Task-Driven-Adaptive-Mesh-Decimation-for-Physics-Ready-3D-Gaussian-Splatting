"""Run Module 5 steps 1-9 in order.

ROI AABB currently comes from roi_config.py (Module 2 stand-in).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mesh_cleaning
import connected_components
import topology_repair
import roi_subroi_mapping
import qem_adaptive_mesh
import mesh_subroi_linking
import adaptive_mesh_selection
import mesh_quality_validation
import collision_mesh_generation


def main():
    mesh_cleaning.main()
    connected_components.main()
    topology_repair.main()
    roi_subroi_mapping.main()
    qem_adaptive_mesh.main()
    mesh_subroi_linking.main()
    adaptive_mesh_selection.main()
    mesh_quality_validation.main()
    collision_mesh_generation.main()
    print("\nMODULE 5 PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
