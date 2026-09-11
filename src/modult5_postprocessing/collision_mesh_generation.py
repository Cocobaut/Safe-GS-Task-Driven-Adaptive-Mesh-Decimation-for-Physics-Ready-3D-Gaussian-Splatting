import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roi_config import (
    COLLISION_DIR,
    COLLISION_MESH,
    COLLISION_REPORT,
    VALIDATED_MESH,
    ensure_dir,
)


def topology_statistics(mesh):
    triangles = np.asarray(mesh.triangles)
    edge_map = {}
    for tri_id, tri in enumerate(triangles):
        v0, v1, v2 = map(int, tri)
        for edge in (
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0))),
        ):
            edge_map.setdefault(edge, []).append(tri_id)

    boundary_edges = 0
    non_manifold_edges = 0
    manifold_edges = 0
    for incident in edge_map.values():
        count = len(incident)
        if count == 1:
            boundary_edges += 1
        elif count == 2:
            manifold_edges += 1
        else:
            non_manifold_edges += 1

    return {
        "edges": len(edge_map),
        "boundary_edges": boundary_edges,
        "manifold_edges": manifold_edges,
        "non_manifold_edges": non_manifold_edges,
        "watertight": boundary_edges == 0 and non_manifold_edges == 0,
    }


def main():
    print("=" * 60)
    print("STEP 9 - COLLISION MESH EXPORT")
    print("=" * 60)
    print("Input mesh:", VALIDATED_MESH)
    print("No extra uniform QEM: keep the validated adaptive mesh.")

    ensure_dir(COLLISION_DIR)
    if not os.path.exists(VALIDATED_MESH):
        raise FileNotFoundError(f"Input mesh not found:\n{VALIDATED_MESH}")

    mesh = o3d.io.read_triangle_mesh(VALIDATED_MESH)
    if mesh.is_empty():
        raise RuntimeError("Input mesh is empty.")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    original = o3d.io.read_triangle_mesh(VALIDATED_MESH)
    original_vertices = len(original.vertices)
    original_triangles = len(original.triangles)
    original_topology = topology_statistics(original)
    final_topology = topology_statistics(mesh)

    valid_mesh = (
        len(mesh.vertices) > 0
        and len(mesh.triangles) > 0
        and final_topology["non_manifold_edges"] == 0
    )

    if not o3d.io.write_triangle_mesh(COLLISION_MESH, mesh):
        raise RuntimeError("Failed to save collision mesh.")

    report = {
        "step": 9,
        "input_mesh": VALIDATED_MESH,
        "output_mesh": COLLISION_MESH,
        "method":
            "Export validated adaptive mesh as collision geometry "
            "(no additional uniform QEM).",
        "input": {
            "vertices": original_vertices,
            "triangles": original_triangles,
            **original_topology,
        },
        "output": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.triangles)),
            **final_topology,
            "polygon_reduction_percent": 0.0,
        },
        "valid": valid_mesh,
    }

    with open(COLLISION_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("Vertices :", len(mesh.vertices))
    print("Triangles:", len(mesh.triangles))
    print("Watertight:", final_topology["watertight"])
    print("Valid:", valid_mesh)
    print("Saved:", COLLISION_MESH)
    print("Report:", COLLISION_REPORT)
    print("\nSTEP 9 COMPLETED")


if __name__ == "__main__":
    main()
